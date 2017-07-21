__author__ = 'smithe3'

import cv2
import numpy as np

doc = open("DataSet.txt", "r")
letters = doc.read()

labels = []
types = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
         "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

for i in range(1, 63):
    for j in range(1, 56):
        labels.append(i)


data = np.array(letters)
labels = np.array(labels)
data = np.float32(data)

knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
ret, result, neighbours, dist = knn.findNearest(letters, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print(accuracy)


