import numpy as np
import os
from Cav_clean import CAV
from FeatureExtractor import FeatureExtractor
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
from ModelWrapper_Clean import ModelWrapper_Clean

def compute_directional_derivatives(gradient, cav):
    '''
    Computes whether the directional derivative is negative or positive.
        Parameters: 
            gradient (np array): np array representing a gradient
            cav (np array): np array representing a cav
        Returns:
            dir_der (boolean): boolean representing whether directional 
            derivative is negative or positive 
    '''
    
    dir_der = np.dot(np.squeeze(gradient), np.squeeze(cav))
    return dir_der < 0

def scoring_tcav(model, cav, folder, class_number, layer_name):
    '''
    Computes a tcav score, given a class, cav, and a layer
        Parameters:
            model (pytorch model): model
            cav (np array): np array representing a cav
            folder (string): name of folder containing images of desired class
            class_number (int): int corresponding to the imagenet classication
            layer_name (np array): name of layer that activations are extracted
                                   from
        Returns:
            score (float): number representing TCAV score
    '''
    print("Scoring TCAV:")

    gradients = []
    files = os.listdir(folder)
    model_wrapper = ModelWrapper_Clean(model, layer_name)
    model_wrapper.eval()
    gradients = []

    for file in tqdm(files[0:250]):
        img = Image.open(folder + "/" + file)
        
        convert_tensor = transforms.ToTensor()
        # creates a function that transforms PIL Image's to Pytorch Tensor's 

        img_as_tensor = convert_tensor(img)
        img_as_tensor = img_as_tensor[None, :, :, :]
        # expands dimensions from [3, 224, 224] to [1, 3, 224, 224]
        assert(img_as_tensor.shape == torch.Size([1, 3, 224, 224]))

        _ = model_wrapper(img_as_tensor)

        gradient = model_wrapper.get_gradients(class_number, layer_name) #TODO: inconsistent
        gradient = gradient.flatten()
        # get and convert gradient from tensor to flattened numpy array 

        gradients.append(gradient)

    gradients = np.array(gradients)
    
    print("Shape of Gradients:", gradients.shape)

    score = np.mean(np.array([compute_directional_derivatives(grad, cav) for grad in gradients]).astype(np.int))

    return score   
    


class TCAV(object):
    def __init__(self, model, class_number, layers, folder):
        '''
        Initialize TCAV object
            Parameters:
                model (pytorch model): pretrained pytorch model trained on imagenet
                class_number (int): int corresponding to the imagenet classication
                layers (np array): name of layer that activations are extracted from
                folder (string): name of folder containing pictures of intended class
        '''
        self.model = model
        self.class_number = class_number
        self.layers = layers
        self.folder = folder
        self.cavs = []
    
    def generate_cavs_sklearn_class(self, concept, randomfiles):
        for layer in self.layers:
            cav_obj = CAV(self.model, concept, randomfiles, layer) #TODO: adapt for multiple players
            cav, accuracy = cav_obj.generate_CAV_from_sklearn_classifier() #TODO: make option to use several functions here
            print("Accuracy of CAV Test Set Scikit-Learn SGDClassifier for layer", layer,  ":", accuracy)
            self.cavs.append(cav) #TODO: adapt for multiple cavs

    def generate_cavs_sklearn_logreg(self, concept, randomfiles):
        for layer in self.layers:
            cav_obj = CAV(self.model, concept, randomfiles, layer) #TODO: adapt for multiple players
            cav, accuracy = cav_obj.generate_CAV_from_sklearn_logreg() #TODO: make option to use several functions here
            print("Accuracy of CAV Test Set Scikit-Learn Logistic Regression for layer", layer,  ":", accuracy)
            self.cavs.append(cav) #TODO: adapt for multiple cavs

    def generate_cavs_pytorch_class(self, concept, randomfiles):
        for layer in self.layers:
            cav_obj = CAV(self.model, concept, randomfiles, layer) #TODO: adapt for multiple players
            cav, _, accuracy = cav_obj.generate_CAV_from_pytorch_classifier() #TODO: make option to use several functions here
            print("Accuracy of CAV Test Set Pytorch Classifier for layer", layer,  ":", accuracy)
            self.cavs.append(cav) #TODO: adapt for multiple cavs

    def return_tcav_score(self, cav, layer): #TODO: allow to put in cavs
        return scoring_tcav(self.model, cav, self.folder, self.class_number, layer)