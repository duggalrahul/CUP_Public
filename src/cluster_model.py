from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA

import src.model
from src.utils import fancy_dendrogram,get_layers,make_idx_dict,get_layer_from_idx,set_layer_to_idx

import numpy as np
import copy
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt


class cluster_model():        
	def __init__(self,model,cluster_args):
		self.model = model   
		self.cluster_layers = cluster_args['cluster_layers']
		self.conv_feature_size = cluster_args['conv_feature_size']
		self.features = cluster_args['features']
		self.channel_reduction = cluster_args['channel_reduction']
		self.use_bias = cluster_args['use_bias']
		self.reshape_exists = cluster_args['reshape_exists']
		self.linkage_method = cluster_args['linkage_method']
		self.criterion = cluster_args['cluster_criterion']  
		self.merge_criterion = cluster_args['merge_criterion']
		self.distance_metric = cluster_args['distance_metric']
		self.distance_threshold = cluster_args['distance_threshold']		
		self.verbose = cluster_args['verbose']
		
		self.compressed_model = None
		self.idx_dict = {}
		
	def cluster_model(self):
		
		self.compressed_model = copy.deepcopy(self.model)
		ctr,self.idx_dict = make_idx_dict(self.compressed_model,-1,[],{})	
		
		for layer_idx in sorted(self.cluster_layers.keys()):
			self.cluster_layer(layer_idx,self.cluster_layers[layer_idx])
				
		return self.compressed_model
		
	def cluster_layer(self,layer_idx,num_drop_nodes):
		# cluster and merge nodes in a layer

		
		layer1 = get_layer_from_idx(self.compressed_model,copy.deepcopy(self.idx_dict),layer_idx)
		layer2 = None
		batchnorm_idx,batchnorm_layer = None,None  
		next_layer_idx = layer_idx       
		while not(isinstance(layer2, nn.Linear) or isinstance(layer2, nn.Conv2d)):
			# add placeholder for batchnorm layer and idx
			next_layer_idx = next_layer_idx + 1
			layer2 = get_layer_from_idx(self.compressed_model,copy.deepcopy(self.idx_dict),next_layer_idx)

			if isinstance(layer2, nn.BatchNorm2d):
				batchnorm_idx = next_layer_idx
				batchnorm_layer = layer2
		# print('weight vs bias before',np.mean(layer1.weight.data.cpu().numpy()),np.mean(layer1.bias.data.cpu().numpy()))
		
		# get feature matrix
		# maybe add batchnorm gamma as a feature
		features = self.get_features([layer1,layer2])

		#decide which nodes to merge                        
		merged_idx = self.get_clustered_idx(features,num_drop_nodes,layer_idx)
		
		# merge nodes        
		layer1_merged,layer2_merged,reshape_info,pruned_batchnorm_layer = self.merge_clusters(features,merged_idx,layer1,layer_idx,layer2,batchnorm_layer)  
		
		# print('weight vs bias after',np.mean(layer1_merged.weight.data.cpu().numpy()),np.mean(layer1_merged.bias.data.cpu().numpy()))
		
				
		set_layer_to_idx(self.compressed_model,copy.deepcopy(self.idx_dict),layer_idx,layer1_merged)
		## add set layer for batchnorm function    
		if self.reshape_exists and (reshape_info != None):
			set_layer_to_idx(self.compressed_model,copy.deepcopy(self.idx_dict),reshape_info[0],reshape_info[1]) 
		if pruned_batchnorm_layer:
			set_layer_to_idx(self.compressed_model,copy.deepcopy(self.idx_dict),batchnorm_idx,pruned_batchnorm_layer)
		
		set_layer_to_idx(self.compressed_model,copy.deepcopy(self.idx_dict),next_layer_idx,layer2_merged)
	   
	def get_features(self,layers):
		features = []

		layer1,layer2 = layers[0],layers[1]

		W1,B1 = layer1.weight.data,layer1.bias.data if layer1.bias is not None else None    
		W2,B2 = layer2.weight.data,layer2.bias.data if layer2.bias is not None else None

		if isinstance(layer1, nn.Linear):
			# then layer2 is guaranteed to be linear
			if self.features == 'incoming':
				features = W1
			elif self.features == 'outgoing':
				features = torch.t(W2)
			elif self.features == 'both':
				features = torch.cat((W1,torch.t(W2)),dim=1)
			else:
				raise Exception("get_features : feature type not implemented.")

		elif isinstance(layer1, nn.Conv2d):
			# layer2 can be linear or conv
			if self.features == 'incoming':
				features = self.get_channel_features(W1,'incoming')
			elif self.features == 'outgoing':
				if isinstance(layer2, nn.Conv2d):
					features = self.get_channel_features(W2,'outgoing')
				elif isinstance(layer2, nn.Linear):
					# Feature map produced by the ith conv filter corresponds to the
					# i*(conv_feature_size^2) to (i+1)*(conv_feature_size^2) columns of the linear layer weights
					
					n_filters = W1.shape[0]
					fm_window = self.conv_feature_size*self.conv_feature_size
					features = torch.stack([torch.norm(torch.stack([W2[:,j] for j in range(f*fm_window,(f+1)*fm_window)]),dim=0) for f in range(n_filters)])
					
					

			elif self.features == 'both': 
				if isinstance(layer2, nn.Conv2d):
					features = torch.cat((self.get_channel_features(W1,'incoming'),self.get_channel_features(W2,'outgoing')),dim=1)
				elif isinstance(layer2, nn.Linear):
					n_filters = W1.shape[0]
					fm_window = self.conv_feature_size*self.conv_feature_size
					out_features = torch.stack([torch.norm(torch.stack([W2[:,j] for j in range(f*fm_window,(f+1)*fm_window)]),dim=0) for f in range(n_filters)])
					   
					features = torch.cat((self.get_channel_features(W1,'incoming'),out_features),dim=1)

			else:
				raise Exception("get_features : feature type not implemented.")
		
		if self.use_bias:
			features = torch.cat((features,B1.reshape(B1.shape[0],1)),dim=1)     

		return features

	def get_channel_features(self,W,f_type):
		'''
			Gets as input a 4-D tensor of weights
			INPUT :
				W : n_out x n_in x K_h x k_w
			OUTPUT :
				f : nout x n_in
		'''
		f = None

		N_out,N_in,K_h,K_w = W.shape

		if f_type == 'incoming':
			f = torch.norm(W.view(N_out,N_in,K_h*K_w),dim=2)
		elif f_type == 'outgoing':
			W_reshape = W.permute(1,0,2,3)
			f = torch.norm(W_reshape.view(N_in,N_out,K_h*K_w),dim=2)
		# f = f / (K_h * K_w)
		return f
	
	
		
	def merge_clusters(self,features,merged_idx,layer1,layer_idx,layer2,batchnorm_layer):

		W1,B1 = layer1.weight.data,layer1.bias.data if layer1.bias is not None else None      
		W2,B2 = layer2.weight.data,layer2.bias.data if layer2.bias is not None else None
		B1_flag,B2_flag = True if B1 is not None else False, True if B2 is not None else False
		reshape_info = None


		if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
			if self.merge_criterion == 'average':
				W1_merged = torch.stack([torch.mean(W1[idx,:],dim=0) for idx in merged_idx])
				W2_merged = torch.stack([torch.mean(W2[:,idx],dim=1) for idx in merged_idx])
				W2_merged = torch.t(W2_merged)
				B1_merged = torch.stack([torch.mean(B1[idx]) for idx in merged_idx]) if B1 is not None else None

			elif self.merge_criterion == 'max_l2_norm':
				# find idx of node representative of clusters
				# a node representative is one which has most energy in its features
				idx_for_cluster = [idx[torch.argmax(torch.norm(features[idx,:],dim=1))] for idx in merged_idx]

				W1_merged = W1[idx_for_cluster,:]
				W2_merged = W2[:,idx_for_cluster]
				B1_merged = B1[idx_for_cluster] if B1 is not None else None

			layer1_merged = nn.Linear(W1_merged.shape[1],W1_merged.shape[0],bias=B1_flag)
			layer2_merged = nn.Linear(W2_merged.shape[1],W2_merged.shape[0],bias=B2_flag)

		elif isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
			
			if self.merge_criterion == 'average':
				# Optional TODO
				raise Exception("merge criterion average not implemented for CONV layers!")

			elif self.merge_criterion == 'max_l2_norm':
				idx_for_cluster = [idx[torch.argmax(torch.norm(features[idx,:],dim=1))] for idx in merged_idx]
				W1_merged = W1[idx_for_cluster,:,:,:]
				W2_merged = W2[:,idx_for_cluster,:,:]
				B1_merged = B1[idx_for_cluster] if B1 is not None else None

			layer1_merged = nn.Conv2d(W1_merged.shape[1],W1_merged.shape[0],W1_merged.shape[2],stride=layer1.stride,padding=layer1.padding,bias=B1_flag)
			layer2_merged = nn.Conv2d(W2_merged.shape[1],W2_merged.shape[0],W2_merged.shape[2],stride=layer2.stride,padding=layer2.padding,bias=B2_flag)

			# print(W1.shape,W2.shape,len(merged_idx),W1_merged.shape,W2_merged.shape)


		elif isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Linear):

			if self.merge_criterion == 'average':
				# Optional TODO
				raise Exception("merge criterion average not implemented for CONV layers!")

			elif self.merge_criterion == 'max_l2_norm':
				idx_for_cluster = [idx[torch.argmax(torch.norm(features[idx,:],dim=1))] for idx in merged_idx]
				W1_merged = W1[idx_for_cluster,:,:,:]

				# W2_merged contains f*(conv_feature_size^2) to (f+1)*(conv_feature_size^2) columns of W2
				# for all f in idx_for_cluster (idx of cluster centers)
				fm_window = self.conv_feature_size*self.conv_feature_size 
				W2_merged = torch.cat([torch.stack([W2[:,j] for j in range(f*fm_window,(f+1)*fm_window)]) for f in idx_for_cluster])
				W2_merged = torch.t(W2_merged)
				B1_merged = B1[idx_for_cluster] if B1 is not None else None
		   
			layer1_merged = nn.Conv2d(W1_merged.shape[1],W1_merged.shape[0],W1_merged.shape[2],stride=layer1.stride,padding=layer1.padding,bias=B1_flag)
			layer2_merged = nn.Linear(W2_merged.shape[1],W2_merged.shape[0],bias=B2_flag)

			# set new reshape layer            
			# reshape_layer = self.get_layer_from_idx(layer_idx)
			if self.reshape_exists:
				reshape_layer = get_layer_from_idx(self.compressed_model,copy.deepcopy(self.idx_dict),layer_idx)
				while not(isinstance(reshape_layer,src.model.Reshape)):
					layer_idx = layer_idx+1
					reshape_layer = get_layer_from_idx(self.compressed_model,copy.deepcopy(self.idx_dict),layer_idx)
				reshape_layer = src.model.Reshape(-1,fm_window*len(idx_for_cluster))
				reshape_info = (layer_idx,reshape_layer)            
		else:
			raise Exception("drop_filters : layer type not implemented!")

		### add batchnorm support here for self.merge_criterion == 'max_l2_norm'
		batchnorm_pruned = None
		if batchnorm_layer:
			if self.merge_criterion == 'max_l2_norm':
				batchnorm_pruned = nn.BatchNorm2d(len(idx_for_cluster))
				batchnorm_pruned.weight.data = batchnorm_layer.weight.data[idx_for_cluster]
				batchnorm_pruned.bias.data = batchnorm_layer.bias.data[idx_for_cluster]
				batchnorm_pruned.running_mean.data = batchnorm_layer.running_mean.data[idx_for_cluster]
				batchnorm_pruned.running_var.data = batchnorm_layer.running_var.data[idx_for_cluster]
			else:
				raise Exception("batchnorm error : merge criterion not implemented")

		layer1_merged.weight.data = W1_merged  
		layer2_merged.weight.data = W2_merged

		if B2 is not None:
			layer2_merged.bias.data = B2

		if self.use_bias:            
			layer1_merged.bias.data = B1_merged   
		
		return layer1_merged,layer2_merged,reshape_info,batchnorm_pruned

	def get_clustered_idx(self,features,num_drop_nodes,layer_idx):
		
		# generate the linkage matrix 
		Z = linkage(features, self.linkage_method,self.distance_metric)
		
		if self.criterion == 'hierarchical_trunc':    
			n_clust = features.shape[0] - num_drop_nodes

			clusters = fcluster(Z,n_clust,criterion='maxclust')
			c, coph_dists = cophenet(Z, pdist(features))

			# print('test',clusters)
			# print('cophonetic coeffcient',c)
		
		if self.criterion == 'hierarchical':            
			# set cut-off to 50
			max_d = self.distance_threshold # max_d as in max_distance
			clusters = fcluster(Z, max_d, criterion='distance')
			
			# calculate truncated dendrogram
			# fancy_dendrogram(
			#     Z,
			#     truncate_mode='lastp',
			#     p=20, # number of merges to show
			#     leaf_rotation=90.,
			#     leaf_font_size=12.,
			#     show_contracted=True,
			#     annotate_above=10,  # useful in small plots so annotations don't overlap
			#     max_d=max_d  # plot a horizontal cut-off line
			# )            
			
		
		merged_idx_d = {}
		for idx,c in enumerate(clusters):
			try:
				merged_idx_d[c].append(idx)
			except:
				merged_idx_d[c] = [idx]

		merged_idx = [value for (key,value) in merged_idx_d.items()]
				
		
		# make a plot of avg feature norm v/s cluster size
		if self.verbose:  

			from matplotlib import rcParams


			plt.tight_layout()

			plt.rc('font', family='serif')
			plt.rc('xtick', labelsize='large')
			plt.rc('ytick', labelsize='large')
			plt.rcParams.update({'font.size': 16})
			rcParams.update({'figure.autolayout': True})              
			merged_idx_c = copy.deepcopy(merged_idx)
			merged_idx_c.sort(key = lambda idx : np.mean([np.linalg.norm(features[i,:],ord=1) for i in idx]),reverse=True)
			
			size,norm = [],[]
			for ctr,idx in enumerate(merged_idx_c):
				size.append(len(idx))
				norm.append(np.mean([np.linalg.norm(features[i,:],ord=1) for i in idx]))
			
			plt.figure(figsize=(6,5))
			plt.title('Average L1 Norm v/s Cluster Size')
			plt.ylabel('Average L1 Norm')
			plt.xlabel('Cluster Size')
			plt.scatter(size,norm,color="blue")
			plt.grid(True)
			plt.savefig(str(layer_idx)+'.png')
			plt.show()
		
		return merged_idx