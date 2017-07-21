__author__ = 'zhangm2'

import cv2
import paper_detection
from imutil import show_img

img = cv2.imread("assets/ipsum.jpg")
paper = paper_detection.get_paper(img)
show_img(paper)

