__author__ = 'zhangm2'

import cv2

def show_img(img):
    cv2.imshow("Main", img)
    cv2.waitKey()