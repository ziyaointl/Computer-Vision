__author__ = 'zhangm2'

import cv2

def show_img(img):
    cv2.imshow("Main", img)
    cv2.waitKey()

def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def otsu(img):
    ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img