import glob
import cv2 as cv
from PIL import Image


def frameSize(path):
    images = Image.open(path)
    print(images)
    frameSize = images.size
    print(f"Taille de l'image : {frameSize}")
    return frameSize

if __name__ == '__main__':
    images = glob.glob('../data/little/frame_0043.jpg')

    for image in images:
        fs=frameSize(image)
        print(fs[0])