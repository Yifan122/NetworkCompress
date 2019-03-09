# Compress networks using PyTorch - Pruning and Quantization

This is a complete training example for Deep Convolutional Networks on ImageNet.
  
Currently, the compression methods based on several techniques below:
  - [Taylor Expansion](https://arxiv.org/abs/1611.06440) (A good summary of this approach can be found [here](https://jacobgil.github.io/deeplearning/pruning-deep-learning)).
  - [Attention Transfer](https://arxiv.org/abs/1612.03928) from paper "Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer" (CVPR 2017)
  - [Knowledge Distillation](https://arxiv.org/abs/1503.02531) from paper "Distilling the Knowledge in a Neural Network" (NIPS 2014)
  - Quantization (grab some code from [here](https://github.com/eladhoffer/convNet.pytorch))

### Dependencies
- Python 3.6.3
- Pytorch 1.0

 
 To clone:
 ```
 git clone https://github.com/Yifan122/network_compress
 ```
 
 example for pruning resnet network:
 ```
 python resnet_prune.py --train_path /home/to/imagenet/taining/dataset --val_path /home/to/imagenet/validation/dataset
```

