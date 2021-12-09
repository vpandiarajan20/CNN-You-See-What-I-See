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
from ModelWrapper import ModelWrapper
# from CAV import preprocess_activations, cav


def compute_directional_derivatives(gradient, cav):
    gradient = gradient.reshape((1,-1))
    dir_der = np.dot(np.squeeze(gradient), np.squeeze(cav)) < 0
    # print(dir_der)
    return dir_der

def scoring_tcav(cav, folder, class_name, layer_name):
    googlenet = models.resnet101(pretrained=True)
    resnet_features = FeatureExtractor(googlenet, layers=[layer_name])
    resnet_features_model_wrapper = ModelWrapper(googlenet, layers=[layer_name])
    activations = []
    activations_model_wrapper = []
    labels = []
    grads  = []
    grads_model_wrapper = []
    for file in os.listdir("zebras_from_kaggle")[0:30]: # need to create variable for zebras
        img = Image.open(folder + "/" + file)
        convert_tensor = transforms.ToTensor()
        dummy_input = convert_tensor(img)
        dummy_input = dummy_input[None, :, :, :]
        features = resnet_features(dummy_input)
        grads.append(resnet_features.get_gradients(463, layer_name)) # needs to be replaced
        features_model_wrapper = resnet_features_model_wrapper(dummy_input)
        grads_model_wrapper.append(resnet_features_model_wrapper.generate_gradients(463, layer_name))

    for x, y in zip(grads, grads_model_wrapper):
        print("START")
        print(x)
        print(":DIFFERENT GRADIENT:")
        print(y)
        print("END")
        print(x-y)
        print("Difference")
        mse = ((x - y)**2).mean(axis=None)
        print(mse)
    
    score = np.mean(np.array([compute_directional_derivatives(grad, cav) for grad in grads]).astype(np.int))
    score_model_wrapper = np.mean(np.array([compute_directional_derivatives(grad, cav) for grad in grads_model_wrapper]).astype(np.int))
    counter = 0
    counter2 = 0
    for grad in grads_model_wrapper:
        if (compute_directional_derivatives(grad, cav)):
            counter += 1
    for grad in grads:
        if (compute_directional_derivatives(grad, cav)):
            counter2 += 1
    print(counter / len(grads))
    print(counter2 / len(grads_model_wrapper))
    print(score)
    print(score_model_wrapper)
    return score