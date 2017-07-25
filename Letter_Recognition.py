__author__ = 'smithe3'

import cv2
import numpy as np

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
print(accuracy)


