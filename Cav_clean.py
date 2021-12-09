import torch
from torch import nn, Tensor
from torch._C import ScriptObject
import torchvision.models as models
from torchvision import transforms
from typing import Dict, Iterable, Callable
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from Classifier import LinearClassifier, train_model
from tqdm import tqdm

import tensorflow as tf #only used for shuffling data

from ModelWrapper_Clean import ModelWrapper_Clean


TEST_PERCENT_SPLIT = 0.2

class CAV(object):
    def __init__(self, model, concept, randomfiles, layer): 
        '''
        CAV class: 
            Parameters: 
                model (pytorch model): pytorch model trained on imagenet 
                concept (string): location of folder with concept that generating CAV for (e.g. Stripes)
                randomfiles (string): location of folder with random files 
                layer (string): name of the layer that activations are extracted from
        '''
        self.model = model
        self.concept = concept
        self.randomfiles = randomfiles
        self.layer = layer

        self.X_train, self.y_train, self.X_test, self.y_test = self.construct_dataset()
        # initializes data set 

    def generate_activations(self, folder):
        '''
        For the layer passed to init, extracts the activations of random images and concept images
            Parameters: 
                folder (string): name of folder that contains the images
        '''

        print("Generating Activations for Images in Folder:", folder)

        files = os.listdir(folder)[0:500]
        model_wrapper = ModelWrapper_Clean(self.model, layers=[self.layer]) #TODO: change name of FeatureExtractor

        all_activations = []

        for file in tqdm(files):
            img = Image.open(folder + "/" + file)
            convert_tensor = transforms.ToTensor()
            # creates a function that transforms PIL Image's to Pytorch Tensor's 

            img_as_tensor = convert_tensor(img)
            img_as_tensor = img_as_tensor[None, :, :, :]
            # expands dimensions from [3, 224, 224] to [1, 3, 224, 224]
            assert(img_as_tensor.shape == torch.Size([1, 3, 224, 224]))

            features = model_wrapper(img_as_tensor)
            activations = torch.flatten(features[self.layer])
            activations = activations.detach().numpy()
            # convert activations from tensor to numpy array 
            all_activations.append(activations)

        all_activations = np.array(all_activations)
        print("Shape of Activations:", all_activations.shape)
        return all_activations

    def construct_dataset(self): 
        '''
        Constructs the dataset to train our linear model. The train data is 
        the activations of this This weights of the classifier will represent CAV
            Parameters: 
            Returns: 
                X_train (np.array): extracted activations from concept and random images for training
                y_train (np.array): labels for concept and random images for training 
                X_test (np.array): extracted activations from concept and random images for testing
                y_test (np.array): labels for concept and random images for testing 
        ----------------------------------------------------------------
        '''

        print("Constructing Dataset to train CAV...")

        concept_activations = self.generate_activations(self.concept)
        random_activations = self.generate_activations(self.randomfiles)
        activations = np.append(concept_activations, random_activations, axis = 0)

        n_conc_activations = concept_activations.shape[0]
        n_rand_activations = random_activations.shape[0]
        labels = np.array([1] * n_conc_activations + [0] * n_rand_activations)
        # creates a numpy array of n_conc_activations 1s and then n_rand_activations 0s

        indices = range(labels.shape[0])
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_activations = tf.gather(activations, shuffled_indices).numpy()
        shuffled_labels = tf.gather(labels, shuffled_indices).numpy()

        idx_split = int((1 -TEST_PERCENT_SPLIT) * labels.shape[0])
        print("Number of train examples:", idx_split)
        print("Number of test examples:", n_conc_activations +  n_rand_activations - idx_split)

        X_train = shuffled_activations[:idx_split]
        y_train = shuffled_labels[:idx_split]
    
        X_test = shuffled_activations[idx_split:]
        y_test = shuffled_labels[idx_split:]


        assert(X_train.shape[0] == y_train.shape[0])
        assert(X_test.shape[0] == y_test.shape[0])

        return X_train, y_train, X_test, y_test


    def generate_CAV_from_pytorch_classifier(self):
        '''
        Generates a concept activation vector by training a pytorch linear classifier 
        to classify random activations from concept activations. The model weights from the classifier 
        is the concept activation vector (normal vector for hyperplace)
            Parameters: 
            Returns: 
                cav (np.array): concept activation vector representing the normal vector to the decision hyperplane
                train_losses (np.array): array of losses as training occurs
                accuracy (float): accuracy of classification on the testing data set
        '''

        print("Generating CAV from Pytorch Linear Classifier...")

        X_train = torch.tensor(self.X_train)
        X_test = torch.tensor(self.X_test)
        y_train = torch.tensor(self.y_train)
        y_test = torch.tensor(self.y_test)
        # generates pytorch tensors
        
        classifer = LinearClassifier(X_train.shape[1])

        n_epochs = 150
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(classifer.parameters(), lr=0.001)

        train_losses = train_model(classifer, criterion, optimizer, X_train, y_train, n_epochs)

        cav = list(classifer.parameters())[0]
        cav = cav.detach().numpy()
        # cav is the model weights  

        logits = classifer(X_test)
        # print(logits)
        pred_labels = (logits > 0).numpy().astype(int)

        pred_equals_labels = (y_test.numpy() == pred_labels)
        accuracy = np.mean(pred_equals_labels)
        # computes accuracy in the test set

        print("Shape of CAV", cav.shape)

        return cav, train_losses, accuracy

    def generate_CAV_from_sklearn_classifier(self):
        '''
        Generates a concept activation vector by training a sklearn linear classifier 
        to classify random activations from concept activations. The model weights from the classifier 
        is the concept activation vector (normal vector for hyperplace)
            Parameters: 
            Returns: 
                cav (np.array): concept activation vector representing the normal vector to the decision hyperplane
                accuracy (float): accuracy of classification on the testing data set

        '''
        print("Generating CAV from Sklearn SGDClassifier...")

        model = SGDClassifier(alpha=0.001)
        model.fit(self.X_train, self.y_train)
    
        cav = np.array(model.coef_)
 
        pred_labels = model.predict(self.X_test) 
        pred_equals_labels = (self.y_test == pred_labels)
        accuracy = np.mean(pred_equals_labels)
        # computes accuracy in the test set

        return cav, accuracy
        
    
    def generate_CAV_from_sklearn_logreg(self):
        '''
        Generates a concept activation vector by training a sklearn logistic regression
        to classify random activations from concept activations. The model weights from the classifier 
        is the concept activation vector (normal vector for hyperplace)
            Parameters: 
            Returns: 
                cav (np.array): concept activation vector representing the normal vector to the decision hyperplane
        '''

        print("Generating CAV from Sklearn LogisticRegression...")

        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
    
        cav = np.array(model.coef_)
 
        pred_labels = model.predict(self.X_test) 
        pred_equals_labels = (self.y_test == pred_labels)
        accuracy = np.mean(pred_equals_labels)
        # computes accuracy in the test set

        return cav, accuracy
