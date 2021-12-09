import numpy as np
import os
from Cav_clean import CAV
from FeatureExtractor import FeatureExtractor
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
from ModelWrapper_Clean import ModelWrapper_Clean

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
    
    def compute_directional_derivatives(self, gradient, cav):
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

    def run_tcav(self, concept, randomfiles, layer, classifier_type):
        '''
        generates the concept activation vector and computes the tcav score
            Parameters: 
                concept (string): location of folder with concept that generating CAV for (e.g. Stripes)
                randomfiles (string): location of folder with random files 
                layer (string): name of the layer that activations are extracted from
                classifier_type (string): The classifier that we want to use
            Returns:
                float representing the tcav score of the CAV
        '''
        cav_obj = CAV(self.model, concept, randomfiles, layer)
        if (classifier_type == "SGD"):
            cav, accuracy = cav_obj.generate_CAV_from_sklearn_classifier()
        elif (classifier_type == "LOG"):
            cav, accuracy = cav_obj.generate_CAV_from_sklearn_logreg()
        elif (classifier_type == "LINEAR"):
            cav, _, accuracy = cav_obj.generate_CAV_from_pytorch_classifier()
        return self.compute_tcav_score(cav, layer)

    def compute_tcav_score(self, cav, layer):
        '''
        computes the tcav score of a given concept activation vector at the given layer
            Parameters: 
                cav (np.array): concept activation vector representing the normal vector to the decision hyperplane
                layer (string): name of the layer that activations are extracted from
            Returns:
                float representing the tcav score of the CAV
        '''
        gradients = []
        files = os.listdir(self.folder)
        model_wrapper = ModelWrapper_Clean(self.model, layer)
        model_wrapper.eval()
        gradients = []

        for file in tqdm(files[0:250]):
            img = Image.open(self.folder + "/" + file)
            
            convert_tensor = transforms.ToTensor()
            # creates a function that transforms PIL Image's to Pytorch Tensor's 

            img_as_tensor = convert_tensor(img)
            img_as_tensor = img_as_tensor[None, :, :, :]
            # expands dimensions from [3, 224, 224] to [1, 3, 224, 224]
            assert(img_as_tensor.shape == torch.Size([1, 3, 224, 224]))

            _ = model_wrapper(img_as_tensor)

            gradient = model_wrapper.get_gradients(self.class_number, layer)
            gradient = gradient.flatten()
            # get and convert gradient from tensor to flattened numpy array 

            gradients.append(gradient)

        gradients = np.array(gradients)
        
        print("Shape of Gradients:", gradients.shape)

        score = np.mean(np.array([self.compute_directional_derivatives(grad, cav) for grad in gradients]).astype(np.int))

        return score   


    def generate_cavs_sklearn_class(self, concept, randomfiles):
        '''
        generates a concept activation vector using the scikit-learn SGD classifier
            Parameters:
                concept:
                randomfiles 
        '''
        for layer in self.layers:
            cav_obj = CAV(self.model, concept, randomfiles, layer) 
            cav, accuracy = cav_obj.generate_CAV_from_sklearn_classifier() 
            print("Accuracy of CAV Test Set Scikit-Learn SGDClassifier for layer", layer,  ":", accuracy)
            self.cavs.append(cav) 

    def generate_cavs_sklearn_logreg(self, concept, randomfiles):
        for layer in self.layers:
            cav_obj = CAV(self.model, concept, randomfiles, layer) 
            cav, accuracy = cav_obj.generate_CAV_from_sklearn_logreg() 
            print("Accuracy of CAV Test Set Scikit-Learn Logistic Regression for layer", layer,  ":", accuracy)
            self.cavs.append(cav) 

    def generate_cavs_pytorch_class(self, concept, randomfiles):
        for layer in self.layers:
            cav_obj = CAV(self.model, concept, randomfiles, layer) 
            cav, _, accuracy = cav_obj.generate_CAV_from_pytorch_classifier() 
            print("Accuracy of CAV Test Set Pytorch Classifier for layer", layer,  ":", accuracy)
            self.cavs.append(cav) 