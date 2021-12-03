import cv2 
import os

def reshape_zebras(dir):
    listing = os.listdir(dir)    
    for file in listing:
        name = dir + "/" + file
        img = cv2.imread(name)
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(name, img)

def main():
    reshape_zebras("zebras_from_kaggles")


if __name__ == '__main__':
    main()