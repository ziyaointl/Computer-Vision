__author__ = 'yangh2'



import cv2

vidCap = cv2.VideoCapture(0)

cv2.ocl.setUseOpenCL(False)


img = cv2.imread("TestImages/Sample.jpg")





hit = []
while 1:
    ret, vidO = vidCap.read()
    hit = [0, 0, 0]
    grey = cv2.cvtColor(vidO, cv2.COLOR_BGR2GRAY)

    res, vid = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY_INV)
    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kp = []
    des = []
    matches = []
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(vid, None)
    
    i = 600
    while i <= 1800:
        kp2, des2 = orb.detectAndCompute(img[0:600, i-600:i], None)
        kp.append(kp2)
        des.append(des2)
        i = i + 600
        matches1 = bfMatcher.match(des1, des2)
        matches1.sort(key = lambda x: x.distance)
        matches.append(matches1)

    # draw matches with distance less than threshold



    for k in range(3):
        for j in range(len(matches[k])):
            if matches[k][j].distance > 10:
                break
            else:
                hit[k] = hit[k] + 1
        img4 = cv2.drawMatches(vid, kp1, img[0:600, 600 * k:600 * k + 600], kp[k], matches[k][:j], None)
    #
    # for j in range(len(matches3)):
    #     if matches3[j].distance > 30:
    #         break
    #     else:
    #         hit[2] = hit[2] + 1
    # img5 = cv2.drawMatches(vid, kp1, img[0:600, 1200:1800], kp4, matches3[:i], None)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vidO, str(hit), (100,100),font, 1, (0,0,255))
    # cv2.imshow("Matchesaa", img[0:600, 1200:1800])
    cv2.imshow("Video", vidO)
    # cv2.imshow("MatchesB", img4)
    # cv2.imshow("MatchesC", img5)
    char = cv2.waitKey(10)
    if chr(char & 0xFF) == 'q':
        # cv2.imwrite("Capture.jpg", vid)
        break
    elif chr(char & 0xFF) == ' ':
        largest = 0
        record = 0
        for i in range(len(hit)):
            print i
            if largest < hit[i]:
                largest = max(hit[i],largest)
                record = i
        print hit
        print "The letter is " + chr(ord('A')+record) + "."

cv2.destroyAllWindows()