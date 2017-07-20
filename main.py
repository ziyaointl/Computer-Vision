import cv2
import numpy

img = cv2.imread("assets/square.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cannyImg = cv2.Canny(grayImg, 100, 200)
lines = cv2.HoughLinesP(cannyImg, 1, numpy.pi/180,
                        threshold = 5,
                        minLineLength = 20, maxLineGap = 10)

for lineSet in lines:
    print lineSet
    for line in lineSet:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]),
                 (255, 0, 0), thickness=3)

cv2.imshow("HoughLines", img)

cv2.waitKey()