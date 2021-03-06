from os import pathsep
import torch
from torch import nn, Tensor
from shutil import copy
import pandas as pd
suraj_path = "/Users/suraj/Documents/Brown/Sophomore/Semester1/CSCI_1470/CNN-You-See-What-I-See/tcav/tcav/tcav_examples/image_models/imagenet/YOUR_FOLDER/broden1_224/broden1_224/index.csv"
vignesh_path = "C:/Users/vpand/Documents/College/Fall 2021/NetDissect/dataset/broden1_224/index.csv"
imgDF = pd.read_csv(suraj_path)
imgDF.texture = imgDF.texture.fillna("-1")
imgDF.texture = imgDF.texture.astype(str)

def getRandomImages():
    randomPaths = imgDF.sample(500)
    for index, img in randomPaths.iterrows():
        if ("295" in img.texture or "276" in img.texture or "356" in img.texture): # this is hardcoded, it needs to be better
            print("one of the wanted textures")
        else:
            path = "/Users/suraj/Documents/Brown/Sophomore/Semester1/CSCI_1470/CNN-You-See-What-I-See/tcav/tcav/tcav_examples/image_models/imagenet/YOUR_FOLDER/broden1_224/broden1_224/images/" + img['image']
            print(path)
            copy(path, "./RandomImages")

def getTextures(x, pathname):
    # x is the numebr of the texture you want to get
    print(imgDF.head())
    rows = imgDF[imgDF.texture.str.contains(x)]
    for index, img in rows.iterrows():
        path = "/Users/suraj/Documents/Brown/Sophomore/Semester1/CSCI_1470/CNN-You-See-What-I-See/tcav/tcav/tcav_examples/image_models/imagenet/YOUR_FOLDER/broden1_224/broden1_224/images/" + img['image']
        print(path)
        try:
            copy(path, "./" + pathname)
        except:
            continue


def main():
    getRandomImages()
    getTextures("295", "Striped")
    getTextures("276", "Dotted")
    getTextures("356", "ZigZag")
    print("Done")

if __name__ == '__main__':
    main()
