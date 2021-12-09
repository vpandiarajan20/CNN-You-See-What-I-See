from numpy.lib.function_base import gradient
import torch
from torch import nn

class ModelWrapper_Clean(nn.Module):
    def __init__(self, model, layers):
        '''
        Initialize Model Wrapper
            Parameters:
                model (pytorch model): pretrained pytorch model
                layers (np array): array of layers to get activations from
        '''
        super().__init__()
        self.model = model
        self.activations = {}

        def hook_function(name):
            '''create hook'''
            def hook(module, input, activation):
                '''create function that saves activations'''
                self.activations[name] = activation
            return hook

        for name, module in self.model._modules.items():
            if name in layers:
                # register the hook
                module.register_forward_hook(hook_function(name))

    def backward_hook(self, grad):
        '''backward hook, saves gradient to class variable'''
        self.gradients = grad

    def get_gradients(self, c, layer_name):
        '''
        registers backward hook, returns gradients
            Parameters:
                c (int): represents class number of image
                layer_name (string): represents layer that we want gradients from
            Returns:
                gradients (np array): gradients out of layer_name
                
        '''
        activation = self.activations[layer_name]
        activation.register_hook(self.backward_hook)
        logit = self.filter_by_class(c)
        gradients = self.backward_pass(logit)
        return gradients

    def backward_pass(self, logits):
        '''
        performs backward pass and returns gradients
            Parameters:
                logits (np array): output of model given a image of intended class
            Returns:
                gradients (np array): gradients out of layer specified earlier
        '''
        logits.backward(torch.ones_like(logits), retain_graph=True)
        gradients = self.gradients.cpu().detach().numpy()
        return gradients


    def filter_by_class(self, c):
        return self.output[:, c]

    def forward(self, x):
        '''
        performs forward pass
            Parameters:
                x (np array): input for model
            Returns:
                output (np array): output of model
        '''
        self.output = self.model(x)
        return self.output