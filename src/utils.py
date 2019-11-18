from sklearn.manifold import TSNE
import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pickle
import torch.nn as nn
import copy


def weights_init(module):
    if isinstance(module, nn.BatchNorm2d):
        # print(module,'batchnorm')
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        # print(module,'conv')
        nn.init.xavier_uniform_(module.weight,gain=nn.init.calculate_gain('relu'))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        # print(module,'linear')
        nn.init.xavier_uniform_(module.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(module.bias, 0)
            

def get_layers(network,all_layers=[]):
    '''
    gets all layers of a network
    '''
    for layer in network.children():
        if type(layer) == nn.Sequential: 
            get_layers(layer,all_layers)
        if list(layer.children()) == []: 
            all_layers.append(layer)
    return all_layers

def make_idx_dict(model,ctr,ary,d):
    for m_idx,m_k in enumerate(model._modules.keys()): 
        n_ary = copy.deepcopy(ary)
        if len(model._modules[m_k]._modules.keys()):
            n_ary.append(m_k)
            ctr,d = make_idx_dict(model._modules[m_k],ctr,n_ary,d)
        else:
            n_ary.append(m_k)            
            ctr = ctr+1
            d[ctr] = n_ary            
    return ctr,d

def get_layer_from_idx(model,idx_ds,idx):
    if len(idx_ds[idx]) == 1:
        return model._modules[idx_ds[idx][0]]
    m_idx = idx_ds[idx].pop(0)
    return get_layer_from_idx(model._modules[m_idx],idx_ds,idx)

def set_layer_to_idx(model,idx_ds,idx,layer):
    if len(idx_ds[idx]) == 1:
        model._modules[idx_ds[idx][0]] = layer
    else:
        m_idx = idx_ds[idx].pop(0)
        set_layer_to_idx(model._modules[m_idx],idx_ds,idx,layer)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def plot_tsne(weight, labels=None, perplexity=40, verbosity=0):
    time_start = time.time()
    tsne = TSNE(n_components=2, init='pca', verbose=verbosity, perplexity=perplexity, n_iter=1000, learning_rate=100)
    tsne_results = tsne.fit_transform(weight)
    exec_time = time.time()-time_start
    print('t-SNE done! Time elapsed: {} seconds'.format(exec_time))
    
    X1,X2 = tsne_results[:,0],tsne_results[:,1]
    x1_min, x1_max = np.min(X1, 0), np.max(X2, 0)
    x2_min, x2_max = np.min(X2, 0), np.max(X2, 0)
    X1 = (X1 - x1_min) / (x1_max - x1_min)
    X2 = (X2 - x2_min) / (x2_max - x2_min)
    
    plt.figure()
    if labels is None:
        plt.plot(X1,X2,'o')     
        plt.title('perplexity: '+str(perplexity))
    else:
        plt.scatter(X1,X2,c=labels, cmap='viridis')
        plt.title('perplexity: '+str(perplexity)+', n_clusters: '+str(max(labels)+1))
    
    
    
    
    return tsne_results


def fancy_dendrogram(*args, **kwargs):
    '''
    function taken from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#comment-12091
    '''
    plt.figure(figsize=(25, 10))
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
