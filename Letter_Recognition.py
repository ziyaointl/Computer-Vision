__author__ = 'smithe3'

import cv2
import numpy as np
import imutil
import char_segmentation

doc = open("assets/DataSet.txt", "r")
#letters = doc.read()

labels = []
rawData = []
data = []
types = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


for i in range(62):
    for j in range(55):
        labels.append(i)

counter = 0
for i in range(62):
    for j in range(55):
        temp = ""
        for k in range(23):
            temp += doc.readline()
            counter = counter+1

        if (len(temp)!=1644):
            print "length " + str(len(temp))
            print "counter position " + str(counter)
            print
        #print len(temp)
        rawData.append(temp)

for i in range(len(rawData)):
    temp = rawData[i].split()
    for j in range(len(temp)):
        temp[j] = int(temp[j])
    #temp = np.array(temp)
    #temp = np.float32(temp)
    #if(len(temp) != 400):
        #print temp
    data.append(temp)


#print data
data = np.array(data)
labels = np.array(labels)
data = np.float32(data)
labels = np.float32(labels)

print labels.shape
print type(labels[0])

print data.shape
print type(data[0][0])

knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
ret, result, neighbours, dist = knn.findNearest(data, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
#print(accuracy)

print "finished training"
'----------------------------------------------------------------------------------------------------------------------'

#src = cv2.imread("assets/Letters/letterE.jpg", 0)

#img = cv2.imread("assets/ipsum.jpg")
img = cv2.imread("assets/Letters/sentences.JPG")
listOfChars, img = char_segmentation.get_chars(img)
image = imutil.getBoundedImg(img, listOfChars[0][2])
gray = imutil.to_gray_scale(image)
#imutil.show_img(image)

print "loaded gray image"

cv2.imshow("orig", gray)
cv2.waitKey()

'''
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
cv2.imshow("histogram", cl1)
cv2.waitKey()
'''

st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
erode = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, st, iterations=3)
#cv2.imshow("erode", erode)
#cv2.waitKey()

width, height = erode.shape

print "morphed image"

if(np.count_nonzero(erode)>(width*height)/2):
    res, thresh = cv2.threshold(erode, 115, 255, cv2.THRESH_BINARY)
    #print thresh
    #cv2.imshow("thresh", thresh)
    # cv2.waitKey()
    resized = cv2.resize(thresh, (20, 20))
else:
    resized = cv2.resize(erode, (20, 20))

#cv2.imshow("resized", resized)
#cv2.waitKey(0)

reshaped = np.reshape(resized, (1, 400))
cv2.imshow("reshaped", reshaped)
cv2.waitKey(0)
print "reshaped image"

retype = np.float32(reshaped)
#nbrs = []

retval, results, neighborResponses, dists = knn.findNearest(retype, k=3)

print
print "The retval is " + str(retval)
print "This is a " + str(types[int(retval)])
print
print results.shape
print retval
print results
print


print "Neighbor responeses: ", neighborResponses
for num in neighborResponses[0]:
    print types[int(num)]

#print dists


