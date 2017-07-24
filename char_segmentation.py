__author__ = 'zhangm2'

import cv2
import paper_detection
import imutil
from imutil import show_img

def get_rows(img):
    # Morph Filter
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, st, iterations=3)
    show_img(img)

    # Find contours
    tempImg = img.copy()
    mgCont, contrs, hier = cv2.findContours(tempImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    validContrs = []
    for i in range(len(contrs)):
        if hier[0][i][3] != -1:
            validContrs.append(contrs[i])

    # Bounding boxes
    for x in range(len(validContrs)):
        validContrs[x] = cv2.boundingRect(validContrs[x])

    return validContrs

def get_chars(img):
    paper = paper_detection.get_paper(img)
    show_img(paper)

    # Convert to gray scale
    img = cv2.cvtColor(paper, cv2.COLOR_RGB2GRAY)

    # Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # OTSU binarize
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

    rowBoundingBoxes = get_rows(img)

    # # Sort contours
    # validContrs, boundingBoxes = imutil.sort_contours(validContrs)

    # Draw bounding boxes
    for boundingBox in rowBoundingBoxes:
        x, y, w, h = boundingBox
        cv2.rectangle(paper, (x, y), (x+w, y+h), (0, 255, 0), 1)
    show_img(paper)

    return validContrs

img = cv2.imread("assets/ipsum.jpg")
get_chars(img)