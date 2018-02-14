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

    #Convert to gray scale
    img = cv2.cvtColor(resizedImg, cv2.COLOR_RGB2GRAY)
    if debug:
        show_img(img)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    if debug:
        show_img(img)

    # Adaptive Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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

def index_of_largest_contour(contours):
    """Return the contour that has the largest area in a list of contours"""
    largest_contour_index = 0
    largest_area = -1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour_index = i
    return largest_contour_index

def get_answer_region_contour(contours, hier, i, img):
    """Recursively find the contour enclosing the answer region
    Returns a contour that occupies at least 20% of the area of its direct parent contour,
    repeat until no such contours can be found

    Keyword arguments:
    contours -- the list of all contours
    hier -- contour tree hierarchy returned by cv2.findCountours
    i -- the index of the current contour candidate
    """

    def is_valid_contour(index):
        """Helper function that verifies whether a contour meets the condition of
        occuping at least 20% of the area of its direct parent contour
        """
        area = cv2.contourArea(contours[index])
        if area / parent_area > 0.2:
            return True

    if debug:
        cv2.drawContours(img, [contours[i]], -1, (0, 255, 0), 3)
        show_img(img)

    # If this contour does not have a child, simply return it
    if hier[i][2] < 0:
        return contours[i]
    # Loop through the children of this contour,
    # find the first contour that occupies at least 20% of the area of contours[i]
    parent_area = cv2.contourArea(contours[i])
    child_index = hier[i][2]
    # If the current contour is valid, set it as the the parent contour and recursively call find()
    if is_valid_contour(child_index):
        return get_answer_region_contour(contours, hier, child_index, img)
    while hier[child_index][0] > -1:
        child_index = hier[child_index][0]
        if is_valid_contour(child_index):
            return get_answer_region_contour(contours, hier, child_index, img)
    # If no suitable contours are found, return the current one
    return contours[i]
