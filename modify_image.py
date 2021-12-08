import cv2 
import os
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import os
import random


def reshape_zebras(dir):
    listing = os.listdir(dir)    
    for file in listing:
        name = dir + "/" + file
        img = cv2.imread(name)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(name, img)

def augument_images(imageFolder):
    folder = imageFolder #'/Users/neilxu/Documents/cs1470/CNN-You-See-What-I-See/Striped2'

    imgFolder = folder.split('/')[-1] #'Striped2'

    directory = os.fsencode(folder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            image = io.imread(imgFolder + "/" + filename)
    
            numTransformations = random.randint(1,5)
            
            tfs = random.sample([1,2,3,4,5], numTransformations)
            
            if 1 in tfs:
                rotated = rotate(image, angle=45, mode = 'wrap')
                io.imsave(imgFolder + "/rotated_" + filename, rotated)
            if 2 in tfs: 
                transform = AffineTransform(translation=(25,25))
                wrapShift = warp(image,transform,mode='wrap')
                io.imsave(imgFolder + "/shifted_" + filename, wrapShift)
            if 3 in tfs: 
                flipUD = np.flipud(image)
                io.imsave(imgFolder + "/flipped_" + filename, flipUD)
            if 4 in tfs: 
                sigma=0.25
                noisyRandom = random_noise(image,var=sigma**2)
                io.imsave(imgFolder + "/noisy_" + filename, noisyRandom)
            if 5 in tfs:
                blurred = gaussian(image,sigma = 2,multichannel=True)
                io.imsave(imgFolder + "/blurry_" + filename, blurred)      
            continue
        else:     
            continue

def main():
    # reshape_zebras("zebras_from_kaggle")
    # augument_images("Striped")
    augument_images("Dotted")


if __name__ == '__main__':
    main()




