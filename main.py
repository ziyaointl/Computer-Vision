from paper_detection import get_paper
from imutil import show_img, to_gray_scale, circularity
from k_means import k_means, contour_center, mean
from string import ascii_uppercase
from custom_exceptions import BubbleDetectionError
import cv2
import numpy as np

DEBUG = True
NUM_ROWS = 13
NUM_COLS = 4
NUM_CHOICES = 4
MIN_CIRCLE_AREA = 400 * 2
MAX_CIRCLE_AREA = 600 * 2.5
ROW_START_POS = 75
ROW_OFFSET = 84
COL_START_POS = 168
COL_OFFSET = 292

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
    # Gaussian Blur
    img = cv2.GaussianBlur(img, (9, 9), 0)
    if DEBUG:
        show_img(img)

    # Adaptive Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 47, 5)
    if DEBUG:
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
    # TODO: Also filter contours by circularity or convexity?
    contrs = [c for c in contrs if cv2.contourArea(c) > MIN_CIRCLE_AREA and cv2.contourArea(c) < MAX_CIRCLE_AREA]
    number_of_bubbles = len(contrs)
    expected_number_of_bubbles = NUM_COLS * NUM_ROWS * NUM_CHOICES
    if number_of_bubbles < int(expected_number_of_bubbles * 0.9):
        raise BubbleDetectionError('Insufficient number of detected bubbles')
    if number_of_bubbles > int(expected_number_of_bubbles * 1.1):
        raise BubbleDetectionError('Too many bubbles were detected')
    if DEBUG:
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
    rows = k_means(cnt_centers, [(0, ROW_START_POS + row * ROW_OFFSET) for row in range(NUM_ROWS)], vertical_distance, img)
    # Verify number of rows
    if len(rows) != NUM_ROWS:
        raise BubbleDetectionError('Wrong number of rows were found')
    # Sort those rows
    rows = sort_dict(rows, lambda x: x[1])
    # Cluster contour centers by columns
    rows = [k_means(row, [(COL_START_POS + COL_OFFSET * col, 0) for col in range(NUM_COLS)], horizontal_distance, img) for row in
            rows]
    # Verify number of columns in each row
    for row in rows:
        if len(row) != NUM_COLS:
            raise BubbleDetectionError('Wrong number of cols were found')
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
    return grid[(question - 1) % NUM_ROWS][(question - 1) // NUM_ROWS]


def get_ans_from_user(question):
    return 'I'
    # TODO: Input verification
    ans = raw_input('Please enter the answer for question ' + str(question) + ': ')
    return ans


def map_number_to_capital_letter(num):
    return ascii_uppercase[num]


# TODO: Replace 'magic numbers' with either constants or expressions


def find_answers(filename):
    img = cv2.imread(filename)
    img = get_paper(img)
    if DEBUG:
        show_img(img)

    img_with_color = img.copy()
    img_gray = to_gray_scale(img_with_color)
    img = pre_process(img_gray)

    contrs = get_bubble_contours(img, img_with_color)
    grid = get_answer_grid(contrs, img_with_color)
    answers = []

    # Loop through the question numbers and append each detected answer to a list
    for question in range(1, NUM_ROWS * NUM_COLS + 1):
        locations = get_question_location(question, grid)
        ans = ''
        if len(locations) != NUM_CHOICES:
            answers.append(get_ans_from_user(question))
            continue
        means = []
        for i in range(NUM_CHOICES):
            # Calculate average brightness of the circle around each bubble center
            mask = np.zeros(img.shape, np.uint8)
            cv2.circle(mask, locations[i], 12, 255, -1)
            # If average brightness is smaller than 100, regard the bubble as filled
            means.append(cv2.mean(img_gray, mask)[0])
        for outlier_index in outliers(means, 3.5):
            # Guard against the situation when three out of four bubbles are filled
            if means[outlier_index] < mean(means[:i] + means[i + 1:]):
                ans += map_number_to_capital_letter(outlier_index)
        answers.append(ans)
    return answers

import doctest
doctest.testmod()