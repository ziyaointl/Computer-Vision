import cv2
from imutil import show_img

debug = True

def contour_center(cnt):
    """Return the centor of the bounding box of a contour"""
    rect = cv2.boundingRect(cnt)
    x = int(rect[0] + .5 * rect[2])
    y = int(rect[1] + .5 * rect[3])
    return x, y
