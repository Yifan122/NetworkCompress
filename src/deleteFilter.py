import numpy as np
import torch
import torch.nn as nn

LAYER = ['layer1', 'layer2', 'layer3', 'layer4']

def create_conv1_mask(index, shape):
    np_mask = np.ones((shape[0], shape[1], shape[2], shape[3]))
    zeros_filter = np.zeros((shape[1], shape[2], shape[3]))
    np_mask[index, :] = zeros_filter
    return torch.tensor(np_mask, dtype=torch.float32, device=torch.device("cuda"))


def creat_bn1_mask(index, shape):
    bn1_mask = np.ones(shape)
    bn1_mask[index] = 0
    return torch.tensor(bn1_mask, dtype=torch.float32, device=torch.device("cuda"))


def create_conv2_mask(index, shape):
    np_mask = np.ones((shape[0], shape[1], shape[2], shape[3]))
    zeros_filter = np.zeros((shape[0], shape[2], shape[3]))
    np_mask[:, index, :, :] = zeros_filter
    return torch.tensor(np_mask, dtype=torch.float32, device=torch.device("cuda"))

def deleteIndividualFilter(conv1, bn1, conv2, index):
    conv1_mask = create_conv1_mask(index, conv1.weight.shape)
    conv1.weight.data.mul_(conv1_mask)

    bn1_mask = creat_bn1_mask(index, bn1.bias.shape)
    bn1.bias.data.mul_(bn1_mask)
    bn1.weight.data.mul_(bn1_mask)
    bn1.running_mean.data.mul_(bn1_mask)
    bn1.running_var.data.mul_(bn1_mask)

    conv2_mask = create_conv2_mask(index, conv2.weight.shape)
    conv2.weight.data.mul_(conv2_mask)

def deleterFilterPerBlock(model, layer, block, filter, blocktype, dev0=0):

    assert blocktype in ['BasicBlock', 'Bottleneck']
    layer = LAYER.index(layer)+1

    with torch.cuda.device(dev0):
        model.cuda()
        layer = layer + 3
        if isinstance(model, nn.DataParallel):
            block = nn.Sequential(*list(model.module.children()))[layer][block]
        else:
            block = nn.Sequential(*list(model.children()))[layer][block]

        if blocktype == 'BasicBlock':
            conv1 = getattr(block, 'conv1')
            bn1 = getattr(block, 'bn1')
            conv2 = getattr(block, 'conv2')
            for index in filter:
                deleteIndividualFilter(conv1, bn1, conv2, index)
        else:
            for i in range(2):
                conv1 = getattr(block, 'conv{}'.format(i+1))
                bn1 = getattr(block, 'bn{}'.format(i+1))
                conv2 = getattr(block, 'conv{}'.format(i+2))
                for index in filter[i]:
                    deleteIndividualFilter(conv1, bn1, conv2, index)
