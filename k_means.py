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


def get_clusters(pts, centroids, distance_fn):
    pairings = {}
    for pt in pts:
        closest_cent = find_closest(pt, centroids, distance_fn)
        if closest_cent in pairings.keys():
            pairings[closest_cent].append(pt)
        else:
            pairings[closest_cent] = [pt]
    return pairings


def mean(lst):
    """Return the arithmetic mean of a list of numbers
    >>> mean([1, 2, 3])
    2
    """
    assert len(lst) > 0, 'List is empty'
    ans = 0
    for num in lst:
        ans += num
    return ans / len(lst)


def get_centroid(pts):
    x = mean([pt[0] for pt in pts])
    y = mean([pt[1] for pt in pts])
    return x, y

