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


# experiment with differnet models 
# experiment w torch vs sklearn libraries for linear model (torch linear classifier, sgdclassifier, logisitic regression)

def preprocess_activations(randomfiles = "RandomImages", concepts = ["Striped"], classID = 340, layer_name="inception5b"):
    googlenet = models.googlenet(pretrained=True)
    resnet_features = FeatureExtractor(googlenet, layers=[layer_name])
    activations = []
    labels = []
    folder_names = np.array()
    folder_names = concepts.append(randomfiles)

    for folder in folder_names:    
        listing = os.listdir(folder)    
        for file in listing:
            img = Image.open(folder + "/" + file)
            convert_tensor = transforms.ToTensor()
            dummy_input = convert_tensor(img)
            dummy_input = dummy_input[None, :, :, :]
            features = resnet_features(dummy_input)
            gradients = resnet_features.get_gradients(340, layer_name)
            newActs = torch.flatten(features[layer_name])
            newActs = newActs.detach().numpy()
            activations.append(newActs)

            if folder == "RandomImages":
                labels.append(0)
            elif folder == "Striped":
                labels.append(1)

    activations = np.array(activations)
    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=1)
    return activations, labels 

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
        print(cav, cav.shape)
        return cav

    else: 
        print("Shape of Train Dataset: ", activations.shape)
        print("Shape of Labels: ", labels.shape)

        activations, labels = preprocess_activations()

        # ---------------------------------------------------
        # Train Linear Classifier
        # ---------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(activations, labels, 
                                            test_size=0.20, random_state=21)

        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

        classifer = LinearClassifier(activations.shape[1])
        n_epochs = 100

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(classifer.parameters(), lr=0.001)

        train_losses = train_model(classifer, criterion, optimizer, X_train, y_train, n_epochs)

        plt.plot(train_losses, label = 'train loss')
        plt.legend()
        plt.show()

        
        p_test = classifer(X_test)
        
        p_test = (p_test > 0).numpy().astype(int)
        
        equals = (y_test.numpy() == p_test)
        accuracy = np.mean(equals)

        print("Accuracy:", accuracy)

        cav = list(classifer.parameters())
        print(cav)

        torch.save(classifer, file_name)

        return cav 


        # model = SGDClassifier(alpha=0.001)
        # model.fit(X_train, y_train)

        # model.evaulate()

        # if len(model.coef_) == 1:
        #     cav = np.array([-model.coef_[0], model.coef_[0]])
        # else: 
        #     cav = -np.array(model.coef_)
        # print(cav)
        # return cav

def main():
    cav = CAV(from_file=True)


if __name__ == '__main__':
    main()
