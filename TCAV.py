import torch
from torch import nn, Tensor
import torchvision.models as models
from typing import Dict, Iterable, Callable
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from Classifier import LinearClassifier, train_model
from FeatureExtractor import FeatureExtractor
# from CAV import preprocess_activations, cav


def compute_directional_derivatives(gradient, cav):
    gradient = gradient.reshape((1,-1))
    dir_der = np.dot(np.squeeze(gradient), np.squeeze(cav)) < 0
    # print(dir_der)
    return dir_der

def scoring_tcav(cav, folder, class_name, layer_name):
    googlenet = models.googlenet(pretrained=True)
    resnet_features = FeatureExtractor(googlenet, layers=[layer_name])
    activations = []
    labels = []
    grads  = []
    for file in os.listdir("zebras_from_kaggle")[0:4]: # need to create variable for zebras
        img = Image.open(folder + "/" + file)
        convert_tensor = transforms.ToTensor()
        dummy_input = convert_tensor(img)
        dummy_input = dummy_input[None, :, :, :]    
        features = resnet_features(dummy_input)
        grads.append(resnet_features.get_gradients(340, layer_name)) # needs to be replaced
    score = np.mean(np.array([compute_directional_derivatives(grad, cav) for grad in grads]).astype(np.int))
    return score