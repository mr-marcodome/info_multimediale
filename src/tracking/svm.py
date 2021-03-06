import numpy as np
import cv2
import csv

def parseIdCSV(ids):
	g = []
	c = []
	k = []
	for i in ids:
		f = open(i+'_depth.csv', 'rb')
		for line in f.readlines():
				vals = []
				line = line.split(";")
				line[-1] = line[-1].replace("\n","")
				if (line[0] == "001"):
					g.append(["062"])
				if (line[0] == "002"):
					g.append(["063"])
				if (line[0] == "003"):
					g.append(["067"])
				if (line[0] == "004"):
					g.append(["064"])
				if (line[0] == "005"):
					g.append(["065"])
				if (line[0] == "006"):
					g.append(["066"])
				if (line[0] == "007"):
					g.append(["063"])
				if (line[0] == "008"):
					g.append(["064"])
				if (line[0] == "009"):
					g.append(["062"])
				if (line[0] == "010"):
					g.append(["066"])
				if (line[0] == "011"):
					g.append(["065"])
				if (line[0] == "012"):
					g.append(["067"])
				line.remove(line[0])
				c.append(line)
				k.append(float(i))
		f.close()

	x = np.array(c, dtype = np.float32)
	l = np.array(k, dtype = np.float32)
	v = np.array(g, dtype = np.float32)

	return x, l, v

ids = ["062","063","064","065","066","067"]

x, y, k = parseIdCSV(ids)

svm = cv2.SVM()
svmparams = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_NU_SVC,  #NU_SVC
                       nu = 0.35,                 # nu = 0.35
                       )
svm.train(x, y, params = svmparams)

u, v, z = parseIdCSV(["014"])
results = svm.predict_all(u)
i = 0
p = len(z)

while(i < p):

	print results[i], z[i]
	i += 1

