import torch
from torch import nn

class ModelWrapper_Clean(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.activations = {}

        def hook_function(name):
            '''create hook'''

            def hook(module, input, activation):
                self.activations[name] = activation

            return hook

        for name, module in self.model._modules.items():
            if name in layers:
                # register the hook
                module.register_forward_hook(hook_function(name))

    def backward_hook(self, grad):
        self.gradients = grad

    def get_gradients(self, c, layer_name):
        activation = self.activations[layer_name]
        activation.register_hook(self.backward_hook)
        logit = self.filter_by_class(c)
        return self.backward_pass(logit)

    def backward_pass(self, logits):
        logits.backward(torch.ones_like(logits), retain_graph=True)
        gradients = self.gradients.cpu().detach().numpy()
        return gradients


    def filter_by_class(self, c):
        return self.output[:, c]

    def forward(self, x):
        self.output = self.model(x)
        return self.output