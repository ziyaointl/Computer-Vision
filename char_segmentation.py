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

    # Sort contours
    validContrs, boundingBoxes = imutil.sort_contours(validContrs, method="top-to-bottom")

    return boundingBoxes

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



    # Find rows
    rowBoundingBoxes = get_rows(img)

    # Put characters in rows
    charsInRows = []
    for row in rowBoundingBoxes:
        a, b, w, h = row
        chars = []
        for validContr in validContrs:
            charBoundingBox = cv2.boundingRect(validContr)
            center = (charBoundingBox[0] + charBoundingBox[2] / 2, charBoundingBox[1] + charBoundingBox[3] / 2)
            if center[0] > a and center[0] < a + w and center[1] > b and center[1] < b + h:
                chars.append(validContr)
        charsInRows.append(chars)


    # Sort characters
    for a in range(len(charsInRows)):
        charsInRows[a], tempBoundingBoxes = imutil.sort_contours(charsInRows[a])

        # Detect character i and merge it (threshold +-2px)
        charBoundingBoxes = []
        b = 0
        lastI = 0
        while b < len(tempBoundingBoxes) - 1:
            currBox = tempBoundingBoxes[b]
            nextBox = tempBoundingBoxes[b + 1]
            if currBox[0] >= nextBox[0] - 2 and currBox[0] <= nextBox[0] + 2:
                x = currBox[0]
                y = min(currBox[1], nextBox[1])
                w = max(currBox[0] + currBox[2], nextBox[0] + nextBox[2]) - x
                h = max(currBox[1] + currBox[3], nextBox[1] + nextBox[3]) - y
                charBoundingBoxes.append((x, y, w, h))
                lastI = b
                b += 2
            else:
                charBoundingBoxes.append(currBox)
                b += 1

        print lastI
        print len(tempBoundingBoxes) - 2
        if lastI != len(tempBoundingBoxes) - 2:
            charBoundingBoxes.append(tempBoundingBoxes[len(tempBoundingBoxes) - 1])
        charsInRows[a] = charBoundingBoxes

    # Draw bounding boxes for rows
    for row in charsInRows:
        for boundingBox in row:
            a, b, w, h = boundingBox
            cv2.rectangle(paper, (a, b), (a+w, b+h), (0, 255, 0), 1)
    show_img(paper)

    for boundingBox in rowBoundingBoxes:
        a, b, w, h = boundingBox
        cv2.rectangle(paper, (a, b), (a+w, b+h), (0, 255, 0), 1)
    show_img(paper)

    return charsInRows

img = cv2.imread("assets/ipsum.jpg")
get_chars(img)