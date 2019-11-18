## CUP: Cluster Pruning for Compressing Deep Neural Networks

This repository contains the code for our upcoming paper.

The intuition behind CUP is as follows - ''Prune layers with many similar filters more aggresively than others''. We achieve this by leveraging hierarchical agglomerative clustering in the following way.

1. Construct features for each filter using the input and output weight connections.
2. Use hierarchical clustering to build tree representations for each layer. Layers with more similarity in filters have shorter tree (e.g. layer p in fig) versus other layers (e.g. layer q).
3. Use a global threshold t to chop all the trees at a height t. This results in a different number of cluster for each layer.
4. Replace each cluster of filters with a representative filter to achieve pruning. 


### Contents
1. [Requirements](#requirements)
2. [Demo](#training)
3. [References](#references)
4. [Contact](#contact)
5. [License](#license)

## Requirements
To run this code, you will require the following softwares.
1. [Anaconda](https://www.anaconda.com/download/#linux) (for python ver 3.6 ) 
2. [Pytorch](https://pytorch.org/) (ver 0.4.1) 
3. [torchvision](https://pypi.org/project/torchvision/0.1.8/) (ver 0.2.0) 


## Demo
1. The final validation accuracy and size of the model can be viewed by opening the corresponding model_dataset.ipynb file. 
    * For example to view Resnet-50 on Imagenet, simply click on Resnet_50_Imagenet.ipynb
    * Within the file, navigate to the section ''Compressing using CUP''
  
2. We also provide the pre-trained and compressed Resnet-50 model.
    * Download from [here](#) and place it in checkpoints/
    * Update the corresponding path in Resnet_50_Imagenet.ipynb and execute the cells in the notebook using shift+enter


## References
To be updated

## Contact
To be updated

## License
To be updated
