__author__ = 'zhangm2'

import cv2

img = cv2.imread("assets/paper.jpg")
img = cv2.resize(img, (0, 0),
           fx = .2, fy = .2)
cv2.imshow("test", img)
cv2.waitKey()