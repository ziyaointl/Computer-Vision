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

filename = "assets/IMG_4949.JPG"

img = cv2.imread(filename)

counter = 0
for col in range(columns):
    for row in range(rows):
        for choice in range(choices):
            cv2.circle(img, (x, y), circleRadius, (0, 255, 0))
            x += shortWidth
        x -= shortWidth * choices
        y += height
        counter += 1
        if (counter == 3):
            y += 1
            counter = 0
    y = initialY
    x += longWidth + shortWidth * (choices - 1)

show_img(img)