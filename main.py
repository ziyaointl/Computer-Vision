from paper_detection import get_paper
from imutil import show_img
from imutil import to_gray_scale
import numpy as np
from imutil import otsu
import cv2

debug = True

filename = "assets/IMG_8083.JPG"
img = cv2.imread(filename)
if debug:
    show_img(img)

img = get_paper(img)
if debug:
    show_img(img)

# Resize paper and convert to grayscale
img = cv2.resize(img, (0, 0), fx = .5, fy = .5)
resizedImg = img
img = to_gray_scale(img)
if debug:
    show_img(img)

#Gaussian Blur
img = cv2.GaussianBlur(img, (9, 9), 0)
if debug:
    show_img(img)

# Adaptive Threshold
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
if debug:
    show_img(img)

# Morphological Filter: Close
# Eliminate small black dots
st = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, st, 1)
if debug:
    show_img(img)

# Find Contours
imgCont, contrs, hier = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# Filter Contours


# Draw Contours
if debug:
    cv2.drawContours(resizedImg, contrs, -1, (255, 0, 0), 3)
    show_img(resizedImg)