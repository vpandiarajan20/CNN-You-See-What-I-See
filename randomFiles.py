import torch
from torch import nn, Tensor
from shutil import copy
import pandas as pd

def main():
    imgDF = pd.read_csv('/Users/neilxu/Documents/cs1470/CNN-You-See-What-I-See/NetDissect-Lite/dataset/broden1_224/index.csv')
    randomPaths = imgDF.sample(500)
    for index, img in randomPaths.iterrows():
        path = "NetDissect-Lite/dataset/broden1_224/images/" + img['image']
        print(path)
        copy(path, "./random_images")
    print("hello")

if __name__ == '__main__':
    main()
