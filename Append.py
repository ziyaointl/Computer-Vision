__author__ = 'smithe3'

import cv2
import numpy as np

with open("DataSet.txt", "w+") as doc:

    letters = []

    for i in range(1, 63):
        for j in range(1,56):
            if(i<10):
                if(j<10):
                    address = "assets/English 2/Hnd/Img/Sample00" + str(i) + "/img00" + str(i) + "-00" + str(j) + ".png"
                    #print address
                    image = cv2.imread(address, 0)
                    #print image
                    resize = cv2.resize(image, (20, 20))
                    reshape = np.reshape(resize, (1, 400))
                    letters.append(reshape)

                else:
                    address = "assets/English 2/Hnd/Img/Sample00" + str(i) + "/img00" + str(i) + "-0" + str(j) + ".png"
                    image = cv2.imread(address, 0)
                    resize = cv2.resize(image, (20, 20))
                    reshape = np.reshape(resize, (1, 400))
                    letters.append(reshape)
            else:
                if(j<10):
                    address = "assets/English 2/Hnd/Img/Sample0" + str(i) + "/img0" + str(i) + "-00" + str(j) + ".png"
                    image = cv2.imread(address, 0)
                    resize = cv2.resize(image, (20, 20))
                    reshape = np.reshape(resize, (1, 400))
                    letters.append(reshape)
                else:
                    address = "assets/English 2/Hnd/Img/Sample0" + str(i) + "/img0" + str(i) + "-0" + str(j) + ".png"
                    image =cv2.imread(address, 0)
                    resize = cv2.resize(image, (20, 20))
                    reshape = np.reshape(resize, (1, 400))
                    letters.append(reshape)
        print "loaded type " + str(i)

    for letter in letters:
        cropped = str(letter)[2:len(str(letter))-2]
        doc.write(cropped)


