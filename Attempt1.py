import torch
from torch import nn, Tensor
import torchvision.models as models
from typing import Dict, Iterable, Callable

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features



def main():
    googlenet = models.googlenet(pretrained=True)
    resnet_features = FeatureExtractor(googlenet, layers=["inception5a"])
    dummy_input = torch.ones(1, 3, 224, 224)
    features = resnet_features(dummy_input)
    print(features)



if __name__ == '__main__':
    main()
