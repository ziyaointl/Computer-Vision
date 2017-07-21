__author__ = 'zhangm2'

import cv2
import numpy
import imutil
from imutil import show_img
from preprocess import preprocess

def get_paper(img):
    #Read in file and resize
    oriImg = img
    ratio = .2
    resizedImg = cv2.resize(oriImg, (0, 0), fx = ratio, fy = ratio)
    show_img(resizedImg)

    #Preprocess
    img = preprocess(resizedImg)

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
    contrs = [finalContr]
    cv2.drawContours(resizedImg, contrs, -1, (0, 255, 0), 3)

    for point in finalContr:
        x = point[0][0]
        y = point[0][1]
        cv2.circle(resizedImg, (x, y), 5, (255, 0, 255))

    show_img(resizedImg)

    #Calculate image dimensions
    height, width, depth = resizedImg.shape
    imgCenter = [[width / 2, height / 2]]

    #Assign each corner
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

    #Perspective Transform
    origPts = numpy.float32([upLeft[0] / ratio, lowLeft[0] / ratio, upRight[0] / ratio, lowRight[0] / ratio])
    newPts = numpy.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])
    mat = cv2.getPerspectiveTransform(origPts, newPts)
    workImg = cv2.warpPerspective(oriImg, mat, (width, height))
    return workImg