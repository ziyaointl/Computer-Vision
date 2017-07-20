__author__ = 'zhangm2'

import cv2

def show_img(img):
    cv2.imshow("Main", img)
    cv2.waitKey()

#Read in file and resize
oriImg = cv2.imread("assets/paper.jpg")
oriImg = cv2.resize(oriImg, (0, 0), fx = .2, fy = .2)

#Median Filter (Blur)
img = cv2.medianBlur(oriImg, 11)
show_img(img)

#Morph Filter
st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, st, iterations=1)
show_img(img)

#Convert to gray scale
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
show_img(img)

#Canny
img = cv2.Canny(img, 100, 200)
show_img(img)

#Find contours
ret,thresh = cv2.threshold(img,127,255,0)
imgCont, contrs, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(oriImg, contrs, -1, (0,255,0), 3)
show_img(oriImg)

#Corners
goodFeats = cv2.goodFeaturesToTrack(img, 20, 0.1, 30)
for goodFeatSets in goodFeats:
    for goodFeatPoints in goodFeatSets:
        cv2.circle(oriImg, (goodFeatPoints[0], goodFeatPoints[1]), 5, (255, 255, 255))
show_img(oriImg)