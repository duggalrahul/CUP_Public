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

import numpy as np
import copy
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt
from src.utils import fancy_dendrogram
import math


import pdb

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		return x.view(self.shape)


class ANN(nn.Module):
	def __init__(self):        
		super(ANN, self).__init__()
		
		self.input = Reshape(-1,784)
		self.fc1 = nn.Linear(784, 500)    
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(500, 300)   
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(300, 10)         
		self.output = nn.LogSoftmax(dim=1)                 
		

	def forward(self, x):
		
		x = self.input(x)
		x = self.fc1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		x = self.relu2(x)
		x = self.fc3(x)
		x = self.output(x)
	
		return x
	
class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.relu1 = nn.ReLU(inplace=True)
		self.maxpool1 = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.relu2 = nn.ReLU(inplace=True)
		self.maxpool2 = nn.MaxPool2d(2,2)
		self.reshape = Reshape(-1,4*4*50)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.relu3 = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(500, 10)
		self.output = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		x = self.reshape(x)
		x = self.fc1(x)
		x = self.relu3(x)
		x = self.fc2(x)
		x = self.output(x)
	   
		return x
 

class VGG(nn.Module):
	def __init__(self, vgg_name, num_output):
		super(VGG, self).__init__()
		self.cnn_layers = self._make_layers(cfg[vgg_name])

		# Initialize weights
		for m in self.cnn_layers:
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()

		self.cnn_layers = nn.Sequential(*self.cnn_layers)
		self.reshape = Reshape(-1,1*1*512)
		self.fc = nn.Linear(512, num_output)        
		self.output = nn.LogSoftmax(dim=1)
		
	def forward(self, x):
		

		x = self.cnn_layers(x)
		x = self.reshape(x)
		x = self.fc(x)
		x = self.output(x)
		return x

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return layers


__all__ = ['resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar', 'resnet56_cifar']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockCifar, self).__init__()
        self.block_gates = block_gates
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNetCifar(nn.Module):

    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.reshape = Reshape(-1,1*1*64)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.output = nn.LogSoftmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.reshape(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.output(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNetCifar(BasicBlockCifar, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar(**kwargs):
    model = ResNetCifar(BasicBlockCifar, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar(**kwargs):
    model = ResNetCifar(BasicBlockCifar, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar(**kwargs):
    model = ResNetCifar(BasicBlockCifar, [9, 9, 9], **kwargs)
    return model


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)
		self.relu2 = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out += self.shortcut(x)
		out = self.relu2(out)

		# out = F.relu(self.bn1(self.conv1(x)))
		# out = self.bn2(self.conv2(out))
		# out += self.shortcut(x)
		# out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)
		self.relu3 = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out = self.conv3(out)
		out = self.bn3(out)
		out += self.shortcut(x)
		out = self.relu3(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU(inplace=True)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.avg_pool = nn.AvgPool2d(4)
		self.reshape = Reshape(-1,1*1*2048)
		self.linear = nn.Linear(512*block.expansion, num_classes)
		self.output = nn.LogSoftmax(dim=1) 

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avg_pool(out)
		out = self.reshape(out)
		out = self.linear(out)
		out = self.output(out)
		return out

def ResNet34(num_classes):
	return ResNet(BasicBlock, [3,4,6,3],num_classes=num_classes)

def ResNet50(num_classes):
	return ResNet(Bottleneck, [3,4,6,3],num_classes=num_classes)
	
    # def _make_layers(self, cfg):
    #     layers = []
    #     in_channels = 3
    #     for x in cfg:
    #         if x == 'M':
    #             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    #         else:
    #             layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
    #                        nn.BatchNorm2d(x),
    #                        nn.ReLU(inplace=True)]
    #             in_channels = x
    #     layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    #     return layers




def load_model(model_type,optim_type,args):
	   
	
	if model_type == 'ann':
		model = ANN()
	elif model_type == 'lenet':
		model = LeNet()
	elif model_type == 'vgg16_bn':
		model = VGG('VGG16',num_output=args.num_output)
	elif model_type == 'resnet56_bn':
		model = resnet56_cifar(num_classes=args.num_output)
	
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = model.to(device)   
	
	if optim_type == 'sgd':
		optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=True)

	checkpoint = torch.load(args.checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	
	print('loading state from epoch {} and test loss {}'.format(epoch,loss))
	
	return model,optimizer

