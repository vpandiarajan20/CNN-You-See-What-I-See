import numpy as np
import os
from FeatureExtractor import FeatureExtractor
from PIL import Image
from torchvision import transforms
import tqdm
import torch

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
    gradient = gradient.reshape((1,-1))
    dir_der = np.dot(np.squeeze(gradient), np.squeeze(cav)) < 0
    # print(dir_der)
    return dir_der

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
    model_wrapper = FeatureExtractor(model, layer_name) #TODO: change name of FeatureExtractor

    gradients = []

    for file in tqdm(files):
        img = Image.open(folder + "/" + file)
        convert_tensor = transforms.ToTensor()
        # creates a function that transforms PIL Image's to Pytorch Tensor's 

        img_as_tensor = convert_tensor(img)
        img_as_tensor = img_as_tensor[None, :, :, :]
        # expands dimensions from [3, 224, 224] to [1, 3, 224, 224]
        assert(img_as_tensor.shape == torch.Size([1, 3, 224, 224]))

        model_wrapper(img_as_tensor)
        gradient = model_wrapper.get_gradients(class_number, layer_name)
        gradient = torch.flatten(gradient)
        gradient = gradient.detach().numpy()
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
                layers (np array): name of layer that activations are extracted
                                   from
                folder (string): name of folder containing pictures of intended class
        '''
        self.model = model
        self.class_number = class_number
        self.layers = layers
        self.folder = folder

    def return_tcav_score(self, cav):
        return scoring_tcav(self.model, cav, self.folder, self.class_number, self.layers)