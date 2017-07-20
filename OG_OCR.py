__author__ = 'smithe3'

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('assets/digits.png')
#print img
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
print train.shape
#print train[0].shape
print test.shape
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print( accuracy )


src = cv2.imread("assets/Numbers/Drawn4.jpg", 0)
cv2.imshow("orig", src)
cv2.waitKey()

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(src)
cv2.imshow("histogram", cl1)
cv2.waitKey()

st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
erode = cv2.morphologyEx(cl1, cv2.MORPH_CLOSE, st, iterations=3)
cv2.imshow("erode", erode)
cv2.waitKey()

width, height = erode.shape
cv2.waitKey()

if(np.count_nonzero(erode)>(width*height)/2):
    res, thresh = cv2.threshold(erode, 115, 255, cv2.THRESH_BINARY_INV)
    print thresh
    cv2.imshow("thresh", thresh)
    cv2.waitKey()
    resized = cv2.resize(thresh, (20, 20))
else:
    resized = cv2.resize(erode, (20, 20))

cv2.imshow("resized", resized)
cv2.waitKey(0)

reshaped = np.reshape(resized, (1, 400))
#cv2.imshow("reshaped", reshaped)
#cv2.waitKey(0)


retype = np.float32(reshaped)
retval, results, neighborResponses, dists = knn.findNearest(retype, k=10)

print
print retval
print results
print neighborResponses
print dists
