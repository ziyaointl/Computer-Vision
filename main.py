from paper_detection import get_paper
from imutil import show_img
from imutil import to_gray_scale
from k_means import k_means, contour_center, mean
from string import ascii_uppercase
import cv2
import numpy as np


debug = True

def standard_deviation(lst):
    m = mean(lst)
    ans = 0
    for num in lst:
        ans += (num - m)**2
    return ans

def median_absolute_deviation(lst):
    """
    >>> median_absolute_deviation([1, 1, 2, 2, 4, 6, 9])
    (2.0, 1.0)
    """
    median = np.median(lst)
    return median, np.median([abs(num - median) for num in lst])

def outliers(lst, thresh=3.5):
    median, mad = median_absolute_deviation(lst)
    outlier_indices = []
    for i in range(len(lst)):
        modified_z_score = (0.6745 * (lst[i] - median)) / mad
        if abs(modified_z_score) >= thresh:
            outlier_indices.append(i)
    return outlier_indices

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


def sort_dict(dictionary, key_fn=lambda x: x):
    """Return a list of values sorted according to the keys of the dictionary
    Optionally, pass in a key function. If passed, the result of key_fn(key) is used for sorting
    """
    return [dictionary[key] for key in sorted(dictionary.keys(), key=key_fn)]


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
    # TODO: Decrease adaptive threshold region and use the colored image to determine mean brightness
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 47, 5)
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
    """
    :param img: a preprocessed image of the answer region
    :param original_img: the original colored image, used for visualization if debug is turned on
    :return: a list of contours of answer bubbles
    """
    if original_img is None:
        original_img = img

    # Find contours
    imgCont, contrs, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours by area
    contrs = [c for c in contrs if cv2.contourArea(c) > 400 * 2 and cv2.contourArea(c) < 600 * 2.25]
    # TODO: Error detection by counting the number of valid contours

    if debug:
        cv2.drawContours(original_img, contrs, -1, (255, 0, 0), 3)
        show_img(original_img)
    return contrs


def get_answer_grid(contrs, img):
    """Group individual bubble locations into questions they belong.
    This 'grid' can be passed to get_question_location(question, grid) to get the bubble locations of a question
    :param contrs: contours of answer bubbles
    :param img: the original colored image, used for visualization if debug is turned on in k_means()
    :return: a 2-dimensional list containing bubble locations, sorted by the order they appear on the image
    """
    # k_means clustering
    # Calculate contour centers
    cnt_centers = [contour_center(cnt) for cnt in contrs]
    # Cluster contour centers by rows
    rows = k_means(cnt_centers, [(0, 75 + row * 84) for row in range(13)], vertical_distance, img)
    # Sort those rows
    rows = sort_dict(rows, lambda x: x[1])
    # Cluster contour centers by columns
    rows = [k_means(row, [(168 + 292 * col, 0) for col in range(4)], horizontal_distance, img) for row in
            rows]
    # Sort those columns
    rows = [sort_dict(row, lambda x: x[0]) for row in rows]
    # Sort individual bubbles in each question
    rows = [[sorted(col, key=lambda x: x[0]) for col in row] for row in rows]
    return rows


def get_question_location(question, grid):
    """Map a question to its location on the grid.
    :param question: question number
    :param grid: the grid returned by get_answer_grid()
    :return: an array containing sorted points, each representing the location of a detected bubble of the requested question
    """
    return grid[(question - 1) % 13][(question - 1) // 13]


def get_ans_from_user(question):
    # TODO: Input verification
    ans = raw_input('Please enter the answer for question ' + str(question) + ': ')
    return ans


def map_number_to_capital_letter(num):
    return ascii_uppercase[num]


# TODO: Replace 'magic numbers' with either constants or expressions


def find_answers(filename):
    img = cv2.imread(filename)
    img = get_paper(img)
    img = cv2.resize(img, (0, 0), fx=.5, fy=.5)
    if debug:
        show_img(img)

    img_with_color = img.copy()
    img = pre_process(img)

    contrs = get_bubble_contours(img, img_with_color)
    grid = get_answer_grid(contrs, img_with_color)
    answers = []

    for question in range(1, 53):
        locations = get_question_location(question, grid)
        ans = ''
        if len(locations) != 4:
            answers.append(get_ans_from_user(question))
            continue
        for i in range(4):
            # Calculate average brightness of the circle around each bubble location
            mask = np.zeros(img.shape, np.uint8)
            cv2.circle(mask, locations[i], 12, 255, -1)
            # If average brightness is smaller than 100, regard the bubble as filled
            if cv2.mean(img, mask)[0] < 100:
                ans += map_number_to_capital_letter(i)
        answers.append(ans)
    return answers

import doctest
doctest.testmod()