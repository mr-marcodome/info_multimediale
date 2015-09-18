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
				line.remove(line[0])
				c.append(line)
				k.append(float(i))
		f.close()

	x = np.array(c, dtype = np.float32)
	l = np.array(k, dtype = np.float32)

	return x, l

ids = ["062","063","064","065","066","067"]

x, y = parseIdCSV(ids)
KN = cv2.KNearest()
#svm = cv2.SVM()

KN.train(x,y)

print "Training completato"

z = np.array([
[1787.0,219.168644288,0,0,19473.5,549.144224524,139,1717.0,1045.395,107,120,105,0,0,0],
[1816.0,220.757341633,0,0,19834.0,550.901583195,1469,415.0,1062.36,123,136,122,0,0,0],
[1741.0,206.998613538,12557.0,464.308655858,6473.5,305.320847869,1421,426.0,1018.485,114,121,96,0,0,0],
[1760.0,211.215536133,12030.0,447.137083054,17045.0,532.274166822,1418,442.0,1029.6,109,200,93,0,0,0],
[1758.0,222.160625054,12619.0,440.450790882,18003.5,544.173661113,1465,388.0,1028.43,91,207,77,0,0,0]
#[1558.0,179.674796069,10216.5,421.722869396,11600.5,429.747254252,1170,439.0,911.43,242,248,255,0,0,0]
#[1726.0,185.874118599,14199.5,486.350286126,14612.5,501.546244144,1407,400.0,1009.71,0,9,13,0,0,0]
],dtype = np.float32)

ret, results, neighbours ,dist = KN.find_nearest(z, 10)
	
print results
print neighbours

