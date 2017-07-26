__author__ = 'zhangm2'

import cv2
import numpy

debug = False

def show_img(img):
    cv2.imshow("Main", img)
    cv2.waitKey()

def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def otsu(img):
    ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def getBoundedImg(img, boundingBox):
    x, y, w, h = boundingBox
    subImg = img[y:y+h, x:x+w]
    if (debug):
        show_img(subImg)
    return subImg

def get_char_simple(img):
    # Find contours
    tempImg = img.copy()
    mgCont, contrs, hier = cv2.findContours(tempImg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours
    validContrs = []
    for i in range(len(contrs)):
        if hier[0][i][3] != -1:
            validContrs.append(contrs[i])

    cv2.drawContours(img, validContrs, -1, (255, 0, 0), thickness=5)

    boundingBox = cv2.boundingRect(validContrs[0])

    return getBoundedImg(img, boundingBox)