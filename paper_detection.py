__author__ = 'zhangm2'

import cv2
import numpy
from imutil import show_img

debug = True

def get_paper(img):
    #Read in file and resize
    oriImg = img
    ratioSmall = .2
    ratioLarge = .5
    resizedImg = cv2.resize(oriImg, (0, 0), fx = ratioSmall, fy = ratioSmall)
    if debug:
        show_img(resizedImg)

    #Median Filter (Blur)
    img = cv2.medianBlur(resizedImg, 11)
    if (debug):
        show_img(img)

    #Morph Filter
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, st, iterations=1)
    if (debug):
        show_img(img)

    #Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if debug:
        show_img(img)

    #Canny
    img = cv2.Canny(img, 50, 200)
    if debug:
        show_img(img)

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

    # Draw Contours & corners
    if debug:
        contrs = [finalContr]
        cv2.drawContours(resizedImg, contrs, -1, (0, 255, 0), 3)

        for point in finalContr:
            x = point[0][0]
            y = point[0][1]
            cv2.circle(resizedImg, (x, y), 5, (255, 0, 255))

        show_img(resizedImg)

    #Calculate image dimensions
    maxX = max(finalContr[0][0][0], finalContr[1][0][0], finalContr[2][0][0], finalContr[3][0][0])
    maxY = max(finalContr[0][0][1], finalContr[1][0][1], finalContr[2][0][1], finalContr[3][0][1])
    minX = min((finalContr[0][0][0], finalContr[1][0][0], finalContr[2][0][0], finalContr[3][0][0]))
    minY = min(finalContr[0][0][1], finalContr[1][0][1], finalContr[2][0][1], finalContr[3][0][1])

    imgCenter = [[(maxX + minX) / 2, (maxY + minY) / 2]]

    #Assign each corner
    lowLeft = imgCenter
    lowRight = imgCenter
    upLeft = imgCenter
    upRight = imgCenter

    for point in finalContr:
        x = point[0][0]
        y = point[0][1]
        if (x >= imgCenter[0][0] and y >= imgCenter[0][1]):
            lowRight = point
        elif (x <= imgCenter[0][0] and y >= imgCenter[0][1]):
            lowLeft = point
        elif (x >= imgCenter[0][0] and y <= imgCenter[0][1]):
            upRight = point
        elif (x <= imgCenter[0][0] and y <= imgCenter[0][1]):
            upLeft = point

    height, width, depth = oriImg.shape
    height = int(ratioLarge * height)
    width = int(ratioLarge * width)

    #Perspective Transform
    origPts = numpy.float32([upLeft[0] / ratioSmall, lowLeft[0] / ratioSmall, upRight[0] / ratioSmall, lowRight[0] / ratioSmall])
    newPts = numpy.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
    mat = cv2.getPerspectiveTransform(origPts, newPts)
    workImg = cv2.warpPerspective(oriImg, mat, (width, height))
    return workImg