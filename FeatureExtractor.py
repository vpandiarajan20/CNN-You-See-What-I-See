
import torch
from torch import nn, Tensor
from typing import Dict, Iterable, Callable
from torchvision import transforms
from PIL import Image
import os

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor):
        _ = self.model(x)
        return self._features
    
    def save_grads(self, grad):
        self.gradients = grad

    def get_gradients(self, className, layerName):
        activations = self._features[layerName]
        activations.register_hook(self.save_grads)
        logits = self.output[:, className]
        logits.backward(torch.ones_like(logits), retain_graph = True)

    def call(self, input):
        self.output = self.model(input)
        return self.output
