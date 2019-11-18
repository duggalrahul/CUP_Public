from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import src.model
from src.utils import get_layers,make_idx_dict,get_layer_from_idx,set_layer_to_idx

import numpy as np
import copy
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt


class prune_model():        
	def __init__(self,model,pruning_args):
		self.model = model
		self.criterion = pruning_args['criterion']
		self.use_bias = pruning_args['use_bias']
		self.prune_layers = pruning_args['prune_layers']
		self.conv_feature_size = pruning_args['conv_feature_size']

		self.pruned_model = None
		self.idx_dict = {}
			
	def prune_model(self,criterion=None):
		'''
		num_drop_nodes : array of length = number of layers. It indicates the number of nodes to drop from each layer
		'''
		if criterion:
			self.criterion = criterion
			
		print('Pruning using : ',self.criterion)
		
		self.pruned_model = copy.deepcopy(self.model)

		ctr,self.idx_dict = make_idx_dict(self.pruned_model,-1,[],{})
		
		for layer_idx in self.prune_layers.keys():
			self.prune_layer(layer_idx,self.prune_layers[layer_idx]) 
			
		return self.pruned_model
		  
	def prune_layer(self,layer_idx,num_drop_nodes):

		layer1 = get_layer_from_idx(self.pruned_model,copy.deepcopy(self.idx_dict),layer_idx)		
		layer2 = None
		next_layer_idx = layer_idx
		batchnorm_idx,batchnorm_layer = None,None       
		while not(isinstance(layer2, nn.Linear) or isinstance(layer2, nn.Conv2d)):			 
			next_layer_idx = next_layer_idx + 1
			layer2 = get_layer_from_idx(self.pruned_model,copy.deepcopy(self.idx_dict),next_layer_idx)
			if isinstance(layer2, nn.BatchNorm2d):
				batchnorm_idx = next_layer_idx
				batchnorm_layer = layer2
		
		features = self.get_features([layer1,layer2])
			
		# print('weight vs bias before',np.mean(layer1.weight.data.cpu().numpy()),np.mean(layer1.bias.data.cpu().numpy()))
		
		#decide neurons to prune
		pruned_node_idx = self.get_pruned_idx(features,num_drop_nodes)   

		#prune selected neurons
		layer1_pruned,layer2_pruned,pruned_batchnorm_layer = self.drop_filters(pruned_node_idx,layer1,layer2,layer_idx,batchnorm_layer)
		
		# print('weight vs bias after',np.mean(layer1_pruned.weight.data.cpu().numpy()),np.mean(layer1_pruned.bias.data.cpu().numpy()))		
		
		set_layer_to_idx(self.pruned_model,copy.deepcopy(self.idx_dict),layer_idx,layer1_pruned)
		if pruned_batchnorm_layer:
			set_layer_to_idx(self.pruned_model,copy.deepcopy(self.idx_dict),batchnorm_idx,pruned_batchnorm_layer)
		set_layer_to_idx(self.pruned_model,copy.deepcopy(self.idx_dict),next_layer_idx,layer2_pruned)
	


	def get_features(self,layers):

		features = []

		# print('test',layers)

		for l in layers:
			W,B = l.weight,l.bias

			if isinstance(l, nn.Linear):
				f = W
			elif isinstance(l, nn.Conv2d):
				# features : flatten input weight kernels.
				# N_out x N_in x K_h x K_w -----> N_out x (N_in*K_h*K_w)
				f = W.reshape([-1,np.prod(list(W.shape)[1:])])
			else:
				raise Exception("get_features: layer not supported!")

			if self.use_bias:
				f = torch.cat((f,B.reshape(B.shape[0],1)),dim=1)

			features.append(f)

		return features
		
	def get_pruned_idx(self,features,num_drop_nodes):

		# f = incoming features of current layer
		f = features[0]
		
		if self.criterion == 'random': 
			pruning_ranked_idx = list(range(0,f.shape[0]))
			random.shuffle(pruning_ranked_idx)
		else:
			if self.criterion == 'l2':            
				f_norm = [(idx,n) for idx,n in enumerate(norm(f.data.cpu().numpy(),axis=1))]
			elif self.criterion == 'l1':            
				f_norm = [(idx,n) for idx,n in enumerate(norm(f.data.cpu().numpy(),ord=1,axis=1))]
			else:
				raise Exception("get_pruned_idx : pruning criterion not implemented!")
			
			f_norm = sorted(f_norm,key=lambda t: t[1])
			pruning_ranked_idx = [t[0] for t in f_norm]                     
				
		pruned_idx = pruning_ranked_idx[:num_drop_nodes]    
						
		return pruned_idx
		
	def drop_filters(self,pruned_node_idx,layer1,layer2,layer_idx,batchnorm_layer=None):

		W1 , B1 = layer1.weight.data , layer1.bias.data if layer1.bias is not None else None
		W2 , B2 = layer2.weight.data , layer2.bias.data if layer2.bias is not None else None
		B1_flag,B2_flag = True if B1 is not None else False, True if B2 is not None else False

		batchnorm_pruned = None

		rem_node_idx = [i for i in range(0,W1.shape[0]) if i not in pruned_node_idx]

		
		if isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear):
			W1_pruned,W2_pruned = W1[rem_node_idx,:],W2[:,rem_node_idx]
			# print(W1.shape,W1_pruned.shape)
			layer1_pruned = nn.Linear(W1_pruned.shape[1], W1_pruned.shape[0],bias=B1_flag)
			layer2_pruned = nn.Linear(W2_pruned.shape[1], W2_pruned.shape[0],bias=B2_flag)

		elif isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d):
			W1_pruned,W2_pruned = W1[rem_node_idx,:,:,:],W2[:,rem_node_idx,:,:]
			layer1_pruned = nn.Conv2d(W1_pruned.shape[1], W1_pruned.shape[0],W1_pruned.shape[2],stride=layer1.stride,padding=layer1.padding,bias=B1_flag)
			layer2_pruned = nn.Conv2d(W2_pruned.shape[1], W2_pruned.shape[0],W2_pruned.shape[2],stride=layer2.stride,padding=layer2.padding,bias=B2_flag)

		elif isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Linear):
			# the ith conv filter corresponds to the
			# i*(conv_feature_size^2) to (i+1)*(conv_feature_size^2) columns of the linear layer weights
			fm_window = self.conv_feature_size*self.conv_feature_size 
			rem_layer2_cols = list(np.concatenate([list(range(col*fm_window,(col+1)*fm_window)) for col in rem_node_idx]))
			W1_pruned,W2_pruned = W1[rem_node_idx,:,:,:],W2[:,rem_layer2_cols]

			reshape_layer = get_layer_from_idx(self.pruned_model,copy.deepcopy(self.idx_dict),layer_idx)
			while not(isinstance(reshape_layer, model.Reshape)):
				layer_idx = layer_idx+1
				reshape_layer = get_layer_from_idx(self.pruned_model,copy.deepcopy(self.idx_dict),layer_idx)

			layer1_pruned = nn.Conv2d(W1_pruned.shape[1], W1_pruned.shape[0],W1_pruned.shape[2],stride=layer1.stride,padding=layer1.padding,bias=B1_flag)
			reshape_layer = model.Reshape(-1,fm_window*len(rem_node_idx))
			layer2_pruned = nn.Linear(W2_pruned.shape[1], W2_pruned.shape[0],bias=B2_flag)

			set_layer_to_idx(self.pruned_model,copy.deepcopy(self.idx_dict),layer_idx,reshape_layer)
		else:
			raise Exception("drop_filters : layer type not implemented!")	

		if batchnorm_layer:
			batchnorm_pruned = nn.BatchNorm2d(len(rem_node_idx))
			batchnorm_pruned.weight.data = batchnorm_layer.weight.data[rem_node_idx]
			batchnorm_pruned.bias.data = batchnorm_layer.bias.data[rem_node_idx]
			batchnorm_pruned.running_mean.data = batchnorm_layer.running_mean.data[rem_node_idx]
			batchnorm_pruned.running_var.data = batchnorm_layer.running_var.data[rem_node_idx]

		layer1_pruned.weight.data = W1_pruned
		layer2_pruned.weight.data = W2_pruned

		if layer2_pruned.bias is not None:
			layer2_pruned.bias.data = B2
		
		if self.use_bias:
			B1_pruned = B1[rem_node_idx]
			layer1_pruned.bias.data = B1_pruned

		return layer1_pruned,layer2_pruned,batchnorm_pruned