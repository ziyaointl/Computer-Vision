from paper_detection import get_paper
from imutil import show_img
from imutil import to_gray_scale
from k_means import k_means, contour_center
import cv2
import numpy as np

debug = True


def horizontal_distance(pt1, pt2):
    """Return the horizontal distance between two points (x1, y1) and (x2, y2)
    >>> horizontal_distance((1, 2),(8, 4))
    7
    """
    return abs(pt2[0] - pt1[0])


def vertical_distance(pt1, pt2):
    """Return the vertical distance between two points (x1, y1) and (x2, y2)
    >>> vertical_distance((1, 2),(8, 11))
    9
    """
    return abs(pt2[1] - pt1[1])


def sort_dict(dict, key_fn=lambda x:x):
    """Return a list of values sorted according to the keys of the dictionary
    Optionally, pass in a key function. If passed, the result of key_fn(key) is used for sorting
    """
    return [dict[key] for key in sorted(dict.keys(), key=key_fn)]


filename = "assets/IMG_0232.JPG"
img = cv2.imread(filename)
if debug:
    show_img(img)

img = get_paper(img)
if debug:
    show_img(img)

# Resize paper and convert to grayscale
img = cv2.resize(img, (0, 0), fx = .5, fy = .5)
resizedImg = img
img = to_gray_scale(img)
if debug:
    show_img(img)

#Gaussian Blur
img = cv2.GaussianBlur(img, (9, 9), 0)
if debug:
    show_img(img)

# Adaptive Threshold
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
if debug:
    show_img(img)

# Morphological Filter: Close
# Eliminate small black dots
st = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, st, 1)
if debug:
    show_img(img)

# Find Contours
imgCont, contrs, hier = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# Filter Contours
contrs = [c for c in contrs if cv2.contourArea(c) > 400 and cv2.contourArea(c) < 600]
print(len(contrs))
rect = contour_center(contrs[0])
# Draw Contours
if debug:
    cv2.drawContours(resizedImg, contrs, -1, (255, 0, 0), 3)
    show_img(resizedImg)

# k_means clustering
# Calculate contour centers
cnt_centers = [contour_center(cnt) for cnt in contrs]
# Cluster contour centers by rows
rows = k_means(cnt_centers, [(0, 50 + row * 56) for row in range(13)], vertical_distance, img=resizedImg)
# Sort those rows
rows = sort_dict(rows, lambda x: x[1])
# Cluster contour centers by columns
rows = [k_means(row, [(112 + 195 * col, 0) for col in range(4)], horizontal_distance, img=resizedImg) for row in rows]
# Sort those columns
rows = [sort_dict(row, lambda x: x[0]) for row in rows]
# Sort individual bubbles in each question
rows = [[sorted(col, key=lambda x: x[0]) for col in row] for row in rows]

