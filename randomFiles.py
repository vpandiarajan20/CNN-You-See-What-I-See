from os import pathsep
import torch
from torch import nn, Tensor
from shutil import copy
import pandas as pd

imgDF = pd.read_csv("C:/Users/vpand/Documents/College/Fall 2021/NetDissect/dataset/broden1_224/index.csv")

def getRandomImages():
    randomPaths = imgDF.sample(500)
    for index, img in randomPaths.iterrows():
        path = "C:/Users/vpand/Documents/College/Fall 2021/NetDissect/dataset/broden1_224/images/" + img['image']
        print(path)
        copy(path, "./RandomImages")

def getTextures(x, pathname):
    # x is the numebr of the texture you want to get
    print(imgDF.dtypes)
    imgDF.texture = imgDF.texture.fillna("-1")
    imgDF.texture = imgDF.texture.astype(str)
    print(imgDF.head())
    rows = imgDF[imgDF.texture.str.contains(x)]
    for index, img in rows.iterrows():
        path = "C:/Users/vpand/Documents/College/Fall 2021/NetDissect/dataset/broden1_224/images/" + img['image']
        print(path)
        copy(path, "./" + pathname)


def main():
    getRandomImages()
    getTextures("295", "Striped")
    getTextures("276", "Dotted")
    getTextures("356", "ZigZag")
    print("Done")

if __name__ == '__main__':
    main()
