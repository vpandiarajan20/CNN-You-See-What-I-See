import torch
from torch import nn, Tensor
import torchvision.models as models
from typing import Dict, Iterable, Callable
from torchvision import transforms
from PIL import Image
import os
import numpy as np

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
    # inception_v3 = models.inception_v3(pretrained=True)
    # print(inception_v3)
    # resnet_features = FeatureExtractor(inception_v3, layers=["Mixed_5b"])
    # dummy_input = torch.ones(1, 3, 224, 224)
    # features = resnet_features(dummy_input)
    # print(features)
    googlenet = models.googlenet(pretrained=True)
    resnet_features = FeatureExtractor(googlenet, layers=["inception5b"])
    activations = np.array([])
    folder_names = np.array(["Dotted", "RandomImages", "Striped"])
    for folder in folder_names:    
        listing = os.listdir(folder)    
        for file in listing:
            img = Image.open(folder + "/" + file)
            convert_tensor = transforms.ToTensor()
            dummy_input = convert_tensor(img)
            dummy_input = dummy_input[None, :, :, :]
            features = resnet_features(dummy_input)
            newActs = torch.flatten(features["inception5b"])
            newActs = newActs.detach().numpy()
            newActs = np.expand_dims(newActs, axis=0)
            activations = np.append(activations, newActs, axis=0)
    print(activations.shape)



if __name__ == '__main__':
    main()
