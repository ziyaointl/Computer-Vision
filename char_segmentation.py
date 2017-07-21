__author__ = 'zhangm2'

import cv2
import paper_detection
import imutil
from imutil import show_img

img = cv2.imread("assets/ipsum.jpg")
paper = paper_detection.get_paper(img)
show_img(paper)

#Convert to gray scale
img = cv2.cvtColor(paper, cv2.COLOR_RGB2GRAY)

#Blur
img = cv2.GaussianBlur(img, (3, 3), 0)

#OTSU binarize
img = imutil.otsu(img)
show_img(img)

# Find contours
tempImg = img.copy()
mgCont, contrs, hier = cv2.findContours(tempImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours
validContrs = []
for i in range(len(contrs)):
    if hier[0][i][3] != -1:
        validContrs.append(contrs[i])
        x,y,w,h = cv2.boundingRect(contrs[i])
        cv2.rectangle(paper, (x, y), (x+w, y+h), (0, 255, 0), 1)

# cv2.drawContours(paper, contrs, -1, (0, 255, 0), 1)
show_img(paper)