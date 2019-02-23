class TaylerExpansionPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}
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

    # def forward(self, x):
    #     self.activations = []
    #     self.gradients = []

    def TaylorExpansion(self):
        print("TO Do")