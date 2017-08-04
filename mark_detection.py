import cv2
import imutil
from imutil import show_img

circleRadius = 10
shortWidth = 20
longWidth = 61
height = 34
x = 265
initialY = 227
y = initialY
rows = 13
columns = 4
choices = 4

filename = "assets/IMG_4970.JPG"

img = cv2.imread(filename)

#To grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Erode
st = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img = cv2.morphologyEx(img, cv2.MORPH_ERODE, st, iterations=1)

#Find contours
ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
imgCont, contrs, hier = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

finalContour = None

#Find answer region
maxArea = img.shape[0] * img.shape[1]
targetArea = 485.0 * 468.0
ratio = targetArea / maxArea

for contr in contrs:
    area = abs(cv2.contourArea(contr))
    currRatio = float(area) / maxArea
    if abs((currRatio - ratio) / ratio) < 0.04:
        if finalContour is None or abs(cv2.contourArea(finalContour)) > area:
            finalContour = contr

#Convert back & display contour
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img, [finalContour], -1, (0, 255, 0))

# Draw circles
# counter = 0
# for col in range(columns):
#     for row in range(rows):
#         for choice in range(choices):
#             cv2.circle(img, (x, y), circleRadius, (0, 255, 0))
#             x += shortWidth
#         x -= shortWidth * choices
#         y += height
#         counter += 1
#         if (counter == 3):
#             y += 1
#             counter = 0
#     y = initialY
#     x += longWidth + shortWidth * (choices - 1)

show_img(img)