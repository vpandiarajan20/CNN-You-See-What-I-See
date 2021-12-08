import torch

class ModelWrapper_Clean():
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.activations = {layer: torch.empty(0) for layer in layers}
