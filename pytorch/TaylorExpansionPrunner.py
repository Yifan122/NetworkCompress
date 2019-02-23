import torch
class TaylerExpansionPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}
        self.reset_act_grad()
        self.filter_ranks = {'layer1': [],
                            'layer2': [],
                            'layer3': [],
                            'layer4': []}
        for name in self.filter_ranks.keys():
            layer = getattr(self.model, name)
            for i in range(len(layer._modules)):
                self.filter_ranks[name].append([])


    def reset_act_grad(self):
        self.activations = {'layer1': [],
                            'layer2': [],
                            'layer3': [],
                            'layer4': []}

        self.gradients = {'layer1': [],
                          'layer2': [],
                          'layer3': [],
                          'layer4': []}

        for name in self.activations.keys():
            layer = getattr(self.model, name)
            for i in range(len(layer._modules)):
                self.activations[name].append([])
                self.gradients[name].append([])

        for name in self.activations.keys():
            layer = getattr(self.model, name)
            for i in range(len(layer._modules)):
                layer._modules[i].conv1.register_forward_hook(self.get_activation, self.activations[name][i])
                layer._modules[i].conv1.register_hook(self.get_grad, self.gradients[name][i])

    def get_activation(self, activation):
        def hook(model, input, output):
            activation = output.detach()
        return hook

    def get_grad(self, gradient):
        def hook(grad):
            gradient = grad
        return hook

    def forward(self, x):
        self.reset_act_grad()
        return self.model(x)

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = \
            torch.sum((activation * grad), dim=0, keepdim=True). \
                sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        # modified by

        # Normalize the rank by the filter dimensions
        values = \
            values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def TaylorExpansion(self):
        print("TO Do")