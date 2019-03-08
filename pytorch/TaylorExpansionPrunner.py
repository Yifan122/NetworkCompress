import torch
import copy
from models.resnet import BasicBlock, Bottleneck

LAYER = ['layer1', 'layer2', 'layer3', 'layer4']


class TaylerExpansionPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}
        self.reset_act_grad()

    def reset_act_grad(self):
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
        self.filter_ranks = copy.deepcopy(self.activations)

        # for name in self.activations.keys():
        #     layer = getattr(self.model, name)
        #     for i in range(len(layer._modules)):
        #         # layer._modules[str(i)].conv1.register_hook(self.get_grad(self.gradients[name][i]))
        #         if isinstance(layer._modules[str(i)], Bottleneck):
        #             layer._modules[str(i)].relu1.register_forward_hook(
        #                 self.get_activation(self.activations[name][i][0]))
        #             layer._modules[str(i)].relu1.register_forward_hook(
        #                 self.get_activation(self.activations[name][i][1]))
        #         else:
        #             layer._modules[str(i)].relu1.register_forward_hook(self.get_activation(self.activations[name][i]))

    def get_activation(self, activation):
        def hook(model, input, output):
            activation = output.detach()
        return hook

    def get_grad(self, gradient):
        def hook(grad):
            gradient = grad.detach()

        return hook

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
                # self.activations[layer_name][module.index] = out[1:]
                for i in range(1, len(out)):
                    self.activations[layer_name][module.index][i-1] = out[i]
                    # out[i].register_hook(lambda grad: self.gradients[layer_name][module.index][i-1] = grad)
                    # self.activations[layer_name][module.index][i-1].register_hook(lambda grad: grad*1)
                    out[i].register_hook(self.get_grad(layer_name, module.index, i-1))

        x = self.model.avgpool(out[0])
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x

    def get_grad(self, layer_name, module_index, index):
        def hook(grad):
            activation = self.activations[layer_name][module_index][index]
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data

            values = \
                values / (activation.size(0) * activation.size(2) * activation.size(3))
            if isinstance(self.values[layer_name][module_index][index], torch.Tensor):
                self.values[layer_name][module_index][index] += values
            else:
                self.values[layer_name][module_index][index] = values
        return hook

    # def compute_rank(self, layer_name, module_index, index):
    #     def hook(self, grad):
    #         activation = self.activations[layer_name][module_index]
    #         values = \
    #             torch.sum((activation * grad), dim=0, keepdim=True). \
    #                 sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
    #         # modified by
    #
    #         # Normalize the rank by the filter dimensions
    #         values = \
    #             values / (activation.size(0) * activation.size(2) * activation.size(3))
    #
    #         # if activation_index not in self.filter_ranks:
    #         #     self.filter_ranks[activation_index] = \
    #         #         torch.FloatTensor(activation.size(1)).zero_().cuda()
    #         #
    #         # self.filter_ranks[activation_index] += values
    #         # self.grad_index += 1
    #         return hook

    def TaylorExpansion(self):
        print("TO Do")
