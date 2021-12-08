from numpy.lib.function_base import gradient
import torch
from torch import nn, Tensor
from torch._C import ScriptObject
import torchvision.models as models
from typing import Dict, Iterable, Callable
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from torchvision.models.googlenet import googlenet
from Classifier import LinearClassifier, train_model
from FeatureExtractor import FeatureExtractor
from ModelWrapper import ModelWrapper
from TCAV import scoring_tcav, compute_directional_derivatives


# experiment with differnet models 
# experiment w torch vs sklearn libraries for linear model (torch linear classifier, sgdclassifier, logisitic regression)

def directional_derivative(classifier, cav, classID = 463, layer_name="inception5b"):
    pass


def preprocess_activations(randomfiles = "RandomImages", concepts = ["Dotted"], classID = 463, layer_name="layer4"):
    googlenet = models.resnet101(pretrained=True)
    resnet_features = FeatureExtractor(googlenet, layers=[layer_name])
    resnet_features_model_wrapper = ModelWrapper(googlenet, layers=[layer_name])
    activations = []
    activations_model_wrapper = []
    labels = []
    concepts.append(randomfiles)

    for folder in concepts:    
        listing = os.listdir(folder)    
        for file in listing[0:50]:
            img = Image.open(folder + "/" + file)
            convert_tensor = transforms.ToTensor()
            dummy_input = convert_tensor(img)
            dummy_input = dummy_input[None, :, :, :]
            features = resnet_features(dummy_input)
            resnet_features_model_wrapper(dummy_input)
            features_model_wrapper = resnet_features_model_wrapper.intermediate_activations
            newActs = torch.flatten(features[layer_name])
            newActs = newActs.detach().numpy()
            newActs_model_wrapper = torch.flatten(features_model_wrapper[layer_name])
            newActs_model_wrapper = newActs_model_wrapper.detach().numpy()
            activations.append(newActs)
            activations_model_wrapper.append(newActs_model_wrapper)
            if folder == "RandomImages":
                labels.append(0)
            elif folder == "Dotted":
                labels.append(1)
            print(img)

    activations = np.array(activations)
    activations_model_wrapper = np.array(activations_model_wrapper)
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=1)
    
    for x, y in zip(activations, activations_model_wrapper):
        mse = ((x - y)**2).mean(axis=None)
        print(mse)
    
    # return activations, labels #, gradients 
    return activations_model_wrapper, labels

def CAV(from_file=False, file_name='linear_classifier_model.pt'):
    # inception_v3 = models.inception_v3(pretrained=True)
    # print(inception_v3)
    # resnet_features = FeatureExtractor(inception_v3, layers=["Mixed_5b"])
    # dummy_input = torch.ones(1, 3, 224, 224)
    # features = resnet_features(dummy_input)
    # print(features)

    # ---------------------------------------------------
    # Extract Activations of Stripes and Random Images and Preprocess
    # ---------------------------------------------------
    if from_file: 
        classifier = torch.load(file_name)
        cav = (list(classifier.parameters())[0]).detach().numpy()
        # print(cav, cav.shape)

        tcav_score = scoring_tcav(cav, "zebras_from_kaggle", 463, "inception5b")
        print("score", tcav_score)
        return cav

    else: 
        
        activations, labels = preprocess_activations()
        print("Shape of Train Dataset: ", activations.shape)
        print("Shape of Labels: ", labels.shape)
        # ---------------------------------------------------
        # Train Linear Classifier
        # ---------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(activations, labels, 
                                            test_size=0.20, random_state=21)

        # X_train = torch.tensor(X_train)
        # X_test = torch.tensor(X_test)

        # y_train = torch.tensor(y_train)
        # y_test = torch.tensor(y_test)

        # classifer = LinearClassifier(activations.shape[1])
        
        # n_epochs = 100
        # criterion = torch.nn.BCEWithLogitsLoss()
        # optimizer = torch.optim.Adam(classifer.parameters(), lr=0.001)

        # train_losses = train_model(classifer, criterion, optimizer, X_train, y_train, n_epochs)

        # plt.plot(train_losses, label = 'train loss')
        # plt.legend()
        # plt.show()

        
        # p_test = classifer(X_test)
        
        # p_test = (p_test > 0).numpy().astype(int)
        
        # equals = (y_test.numpy() == p_test)
        # accuracy = np.mean(equals)

        # print("Accuracy:", accuracy)

        # cav = list(classifer.parameters())[0]
        # cav = cav.detach().numpy()
        # print(cav)

        # tcav_score = scoring_tcav(cav, "zebras_from_kaggle", 463, "layer4")
        # print("score", tcav_score)

        # torch.save(classifer, file_name)

        # return cav 


        model = SGDClassifier(alpha=0.001)
        model.fit(X_train, y_train)

        #model.evaulate()

        # if len(model.coef_) == 1:
        #     cav = np.array([-model.coef_[0], model.coef_[0]])
        # else: 
        cav = -np.array(model.coef_)
        print(cav)
        tcav_score = scoring_tcav(cav, "zebras_from_kaggle", 463, "layer4")
        print("score", tcav_score)
        return cav

def main():
    # googlenet = models.resnet101(pretrained=True)
    # for file in os.listdir("zebras_from_kaggle")[0:50]: # need to create variable for zebras
    #     img = Image.open("zebras_from_kaggle" + "/" + file)
    #     convert_tensor = transforms.ToTensor()
    #     dummy_input = convert_tensor(img)
    #     dummy_input = dummy_input[None, :, :, :]
    #     probs = googlenet(dummy_input)
    #     print(probs.shape)
    #     print(np.argmax(np.squeeze(probs.cpu().detach().numpy())))
    # print(googlenet)
    cav = CAV(from_file=False)


if __name__ == '__main__':
    main()
