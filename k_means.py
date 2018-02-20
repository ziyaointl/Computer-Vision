import cv2
from imutil import show_img

debug = True

def contour_center(cnt):
    """Return the centor of the bounding box of a contour"""
    rect = cv2.boundingRect(cnt)
    x = int(rect[0] + .5 * rect[2])
    y = int(rect[1] + .5 * rect[3])
    return x, y


def find_closest(pt, centroids, distance_fn):
    """Return the centroid closest to pt using the distance measured by distance_fn(pt)
    >>> find_closest((0, 0), [(0, 1), (0, 2)], horizontal_distance)
    (0, 1)
    """
    return min(centroids, key=lambda x: distance_fn(x, pt))
