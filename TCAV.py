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
from CAV import preprocess_activations, cav



def compute_directional_derivatives(gradients, cav, classID = 340, layer_name="inception5b"):
    return np.dot(gradients)