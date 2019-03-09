import torch
import copy
import numpy as np
from models.resnet import BasicBlock, Bottleneck
from src.deleteFilter import deleterFilterPerBlock
import torch.optim as optim
from config import config


LAYER = ['layer1', 'layer2', 'layer3', 'layer4']


class TaylerExpansionPrunner:
    def __init__(self, model, blocktype):
        self.model = model
        self.reset()
        self.blocktype = blocktype

    def reset(self):
        self.activations = {'layer1': [],
                            'layer2': [],
                            'layer3': [],
                            'layer4': []}

        for name in self.activations.keys():
            layer = getattr(self.model, name)
            for i in range(len(layer._modules)):
                # self.activations[name].append([])
                if isinstance(layer._modules[str(i)], BasicBlock):
                    self.activations[name].append([])
                else:
                    self.activations[name].append([[], []])

        self.values = copy.deepcopy(self.activations)
        self.filter_delete = copy.deepcopy(self.activations)
        self.filter_important = copy.deepcopy(self.activations)


    def forward(self, x):

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        out = [x]
        for layer_name in LAYER:
            layer = getattr(self.model, layer_name)
            for _, (name, module) in enumerate(layer._modules.items()):
                out = module(out[0], True)

                for i in range(1, len(out)):
                    self.activations[layer_name][module.index][i - 1] = out[i]
                    out[i].register_hook(self.get_grad(layer_name, module.index, i - 1))

        x = self.model.avgpool(out[0])
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x

    def get_grad(self, layer_name, module_index, index):
        def hook(grad):
            activation = self.activations[layer_name][module_index][index]

            # Calculate the first order Taylor expansion
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

            # Nomalize
            values = \
                values / (activation.size(0) * activation.size(2) * activation.size(3))
            if isinstance(self.values[layer_name][module_index][index], torch.Tensor):
                self.values[layer_name][module_index][index] += values
            else:
                self.values[layer_name][module_index][index] = values

        return hook

    def get_min_taylor_filter(self):
        smallest_index = ['layer1', 0, 0]
        smallest_filters = []
        smallest_matitude = 1
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for i in range(len(self.values[layer_name])):
                for j in range(len(self.values[layer_name][i])):
                    # print(type(self.values[layer_name][i][j]))
                    # print(self.values[layer_name][i][j])
                    if isinstance(torch.Tensor, type(self.values[layer_name][i][j])):
                        self.values[layer_name][i][j] = self.values[layer_name][i][j].numpy()

                    value = self.values[layer_name][i][j]
                    # should be largest or smallest?
                    value[self.filter_delete[layer_name][i][j]] = 5
                    idx = np.argpartition(value, 3)
                    if value[idx[2]] < smallest_matitude:
                        smallest_matitude = value[idx[2]]
                        smallest_index = [layer_name, i, j]
                        smallest_filters = idx[:2]


        self.filter_delete[smallest_index[0]][smallest_index[1]][smallest_index[2]] += smallest_filters.numpy().tolist()
        # print(self.filter_delete)
        # for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        #     for i in range(len(self.values[layer_name])):
        #         for j in range(len(self.values[layer_name][i])):
        #             if self.filter_delete[layer_name][i][j]:
        #                 print(self.values[layer_name][i][j])


    def deleteFilter(self):
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for i in range(len(self.filter_delete[layer_name])):
                deleterFilterPerBlock(self.model, layer_name, i, self.filter_delete[layer_name][i], self.blocktype)

    def calculateTaylor(self, dataloader, criterion, epoch=1, print_freq=200, exit_point = config.channel_exit):

        optimizer = optim.SGD(self.model.parameters(), config.lr,
                              momentum=config.momentum)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module

        self.model.train()
        self.model.cuda()

        for _ in range(epoch):

            for i, (input, target) in enumerate(dataloader):
                input = input.cuda()
                target = target.cuda()

                output = self.forward(input)
                loss = criterion(output, target)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

                if i % print_freq == 0:
                    print_message = 'Epoch: [{0}][{1}/{2}]\t' .format(
                                    epoch, i, len(dataloader))
                    print(print_message)

                if i == exit_point:
                    return None

