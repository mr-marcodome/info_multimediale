import numpy as np
import cv2
import csv




def parseIdCSV(ids):

	c = []
	k = []
	for i in ids:
		f = open(i+'_depth.csv', 'rb')
		for line in f.readlines():
				vals = []
				line = line.split(";")
				line[-1] = line[-1].replace("\n","")
				c.append(line)
				k.append(float(i))
		f.close()

	x = np.array(c, dtype = np.float32)
	l = np.array(k, dtype = np.float32)

	return x, l

ids = ["062","063","064","065","066","067"]

x, y = parseIdCSV(ids)
svm = cv2.SVM()
print y
svm.train(x,y)

svm.save("training.xml")

print "Training completato"
