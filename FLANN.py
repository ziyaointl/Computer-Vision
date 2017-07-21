import cv2

FLANN_INDEX_LSH = 6
indexParams= dict(algorithm = FLANN_INDEX_LSH,
                  table_number = 6, # 12
                  key_size = 12,     # 20
                  multi_probe_level = 1) #2
searchParams = dict(checks=50)
img1 = cv2.imread("assets/hatch.jpg")

orb = cv2.ORB_create()
flanner = cv2.FlannBasedMatcher(indexParams, searchParams)

kp1, des1 = orb.detectAndCompute(img1, None)

#Video Capture
vidCap = cv2.VideoCapture(0)

count = 0
while True:
    ret, img2 = vidCap.read()
    kp2, des2 = orb.detectAndCompute(img2, None)
    matches = flanner.match(des1, des2)
    matches.sort(key=lambda x: x.distance)  # sort by distance
    # draw matches with distance less than threshold
    for i in range(len(matches)):
        if matches[i].distance > 30.0:
            break
    img2 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:i], None)
    cv2.imshow("Matches", img2)

    x = cv2.waitKey(20)
    ch = chr(x & 0xFF)
    if ch == 'q':
        break


cv2.destroyAllWindows()
vidCap.release()