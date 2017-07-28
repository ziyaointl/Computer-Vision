__author__ = 'smithe3'

import cv2
import numpy as np
import imutil
import char_segmentation

doc = open("assets/DataSet.txt", "r")

labels = []
rawData = []
data = []
string = []
types = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
         "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


for i in range(62):
    for j in range(55):
        labels.append(i)

counter = 0
for i in range(62):
    for j in range(55):
        temp = ""
        for k in range(23):
            temp += doc.readline()
            counter += 1

        if len(temp) != 1644:
            print "length " + str(len(temp))
            print "counter position " + str(counter)

        rawData.append(temp)

for i in range(len(rawData)):
    temp = rawData[i].split()
    for j in range(len(temp)):
        temp[j] = int(temp[j])
    data.append(temp)


data = np.array(data)
labels = np.array(labels)
data = np.float32(data)
labels = np.float32(labels)

knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
ret, result, neighbours, dist = knn.findNearest(data, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000


print "--------------------finished training--------------------"

string = ""

#src = cv2.imread("assets/Letters/letterE.jpg", 0)
#img = cv2.imread("assets/ipsum.jpg")

img = cv2.imread("assets/exampleText.JPG")
listOfChars, img, space = char_segmentation.get_chars(img)

for j in range(len(listOfChars)):
    for i in range(len(listOfChars[j])):
        image = imutil.getBoundedImg(img, listOfChars[j][i])
        gray = imutil.to_gray_scale(image)

        st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        erode = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, st, iterations=3)


        width, height = erode.shape


        if(np.count_nonzero(erode)>(width*height)/2):
            res, thresh = cv2.threshold(erode, 115, 255, cv2.THRESH_BINARY)

            resized = cv2.resize(thresh, (20, 20))
        else:
            resized = cv2.resize(erode, (20, 20))

        reshaped = np.reshape(resized, (1, 400))
        retype = np.float32(reshaped)
        retval, results, neighborResponses, dists = knn.findNearest(retype, k=3)

        #print "This is a " + str(types[int(retval)])
        string += str(types[int(retval)])
        if i != len(listOfChars[j])-1:
            if space[j][i] == 1:
                string += ' '

    string += "\n"
print string


