__author__ = 'zhangm2'

import cv2

def show_img(img):
    cv2.imshow("Main", img)
    cv2.waitKey()

def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)