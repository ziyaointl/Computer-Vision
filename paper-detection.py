__author__ = 'zhangm2'

import cv2
import numpy

def show_img(img):
    cv2.imshow("Main", img)
    cv2.waitKey()

#Read in file and resize
oriImg = cv2.imread("assets/IMG-4860.JPG")
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

# #Line detection
# lines = cv2.HoughLinesP(img, 1, numpy.pi/180,
#                         threshold = 5,
#                         minLineLength = 20, maxLineGap = 10)
# for lineSet in lines:
#     for line in lineSet:
#         cv2.line(oriImg, (line[0], line[1]), (line[2], line[3]),
#                  (255, 255, 0), thickness=5)
#
# show_img(oriImg)

#Find contours
ret, thresh = cv2.threshold(img, 127, 255, 0)
imgCont, contrs, hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

finalContr = None
contrArea = 0

for x in range(len(contrs)):
    epsilon = 0.1*cv2.arcLength(contrs[x], True)
    approx = cv2.approxPolyDP(contrs[x], epsilon, True)
    if (cv2.isContourConvex(approx)):
        area = abs(cv2.contourArea(approx))
        if area > contrArea and len(approx) == 4:
            finalContr = approx
            contrArea = area

imgCenter = [[oriImg.shape[0] / 2, oriImg.shape[1] / 2]]

lowLeft = imgCenter
lowRight = imgCenter
upLeft = imgCenter
upRight = imgCenter

for point in finalContr:
    x = point[0][0]
    y = point[0][1]
    if (x >= lowRight[0][0] and y >= lowRight[0][1]):
        lowRight = point
    elif (x <= lowLeft[0][0] and y >= lowLeft[0][1]):
        lowLeft = point
    elif (x >= upRight[0][0] and y <= upRight[0][1]):
        upRight = point
    else:
        upLeft = point

contrs = [finalContr]
cv2.drawContours(oriImg, contrs, -1, (0, 255, 0), 3)

for point in finalContr:
    x = point[0][0]
    y = point[0][1]
    cv2.circle(oriImg, (x, y), 5, (255, 0, 255))
show_img(oriImg)

# #Corners
# goodFeats = cv2.goodFeaturesToTrack(img, 20, 0.1, 30)
# for goodFeatSets in goodFeats:
#     for goodFeatPoints in goodFeatSets:
#         cv2.circle(oriImg, (goodFeatPoints[0], goodFeatPoints[1]), 5, (255, 255, 255))
# show_img(oriImg)