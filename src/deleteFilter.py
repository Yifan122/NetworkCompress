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


def deleterFilter(model, layer, block, filter, dev0=0):

    # layer from 1-4
    # block 0 or 1
    layer = LAYER.index(layer)+1

    with torch.cuda.device(dev0):
        model.cuda()
        layer = layer + 3
        if isinstance(model, nn.DataParallel):
            block = nn.Sequential(*list(model.module.children()))[layer][block]
        else:
            block = nn.Sequential(*list(model.children()))[layer][block]

        for index in filter:
            conv1_mask = create_conv1_mask(index, block.conv1.weight.shape)
            block.conv1.weight.data.mul_(conv1_mask)

            bn1_mask = creat_bn1_mask(index, block.bn1.bias.shape)
            block.bn1.bias.data.mul_(bn1_mask)
            block.bn1.weight.data.mul_(bn1_mask)
            block.bn1.running_mean.data.mul_(bn1_mask)
            block.bn1.running_var.data.mul_(bn1_mask)

            conv2_mask = create_conv2_mask(index, block.conv2.weight.shape)
            block.conv2.weight.data.mul_(conv2_mask)

    return model
