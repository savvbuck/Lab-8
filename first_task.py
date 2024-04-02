import numpy as np
import cv2


def image_processing(img_adress='images/variant-7.jpg'):
    image = cv2.imread(img_adress)
    flipped = cv2.flip(cv2.flip(image, 1), 0) #flipped horizontally, then vertically
    cv2.imshow('flipped_image', flipped)
    cv2.waitKey(0)

if __name__ == '__main__':
    image_processing()
