__author__ = 'zhangm2'

import cv2
import imutil
from imutil import show_img

def preprocess(img):
    #Convert to gray scale
    img = imutil.to_gray_scale(img)
    show_img(img)

    #Median Filter (Blur)
    img = cv2.medianBlur(img, 11)
    show_img(img)

    #Morph Filter
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, st, iterations=1)
    show_img(img)

    #Canny
    img = cv2.Canny(img, 100, 200)
    show_img(img)

    return img