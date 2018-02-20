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


def pre_process(img):
    """Preprocess img for contour recognition"""
    # Convert to grayscale
    img = to_gray_scale(img)
    if debug:
        show_img(img)

    # Gaussian Blur
    img = cv2.GaussianBlur(img, (9, 9), 0)
    if debug:
        show_img(img)

    # Adaptive Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    if debug:
        show_img(img)

    # Morphological Filter: Close
    # Eliminate small black dots
    st = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, st, 1)
    if debug:
        show_img(img)

    return img

def get_bubble_contours(img, original_img=None):
    if original_img is None:
        original_img = img
    # Find Contours
    imgCont, contrs, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Filter Contours
    contrs = [c for c in contrs if cv2.contourArea(c) > 400 and cv2.contourArea(c) < 600]
    # TODO: Error detection by counting the number of valid contours
    if debug:
        cv2.drawContours(original_img, contrs, -1, (255, 0, 0), 3)
        show_img(original_img)
    return contrs

def get_answer_grid(contrs, img):
    # k_means clustering
    # Calculate contour centers
    cnt_centers = [contour_center(cnt) for cnt in contrs]
    # Cluster contour centers by rows
    rows = k_means(cnt_centers, [(0, 50 + row * 56) for row in range(13)], vertical_distance, img)
    # Sort those rows
    rows = sort_dict(rows, lambda x: x[1])
    # Cluster contour centers by columns
    rows = [k_means(row, [(112 + 195 * col, 0) for col in range(4)], horizontal_distance, img) for row in
            rows]
    # Sort those columns
    rows = [sort_dict(row, lambda x: x[0]) for row in rows]
    # Sort individual bubbles in each question
    rows = [[sorted(col, key=lambda x: x[0]) for col in row] for row in rows]
    return rows

def get_question_location(question, grid):
    """Maps the question number to its location on the grid.
    Returns an array containing sorted points,
    each point representing the location of a detected bubble of the requested question"""
    return grid[(question - 1) % 13][(question - 1) // 13]

filename = "assets/IMG_0232.JPG"

img = cv2.imread(filename)
img = get_paper(img)
img = cv2.resize(img, (0, 0), fx=.5, fy=.5)
if debug:
    show_img(img)

img_with_color = img.copy()
img = pre_process(img)

contrs = get_bubble_contours(img, img_with_color)
grid = get_answer_grid(contrs, img_with_color)

for question in range(1, 53):
    locations = get_question_location(question, grid)
    for bubble in locations:
        mask = np.zeros(img.shape, np.uint8)
        cv2.circle(mask, bubble, 8, 255, -1)
        print cv2.mean(img, mask)
    print("--------------------" + str(question))

import doctest
doctest.testmod()