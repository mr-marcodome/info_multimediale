import numpy as np
import cv2

svm = cv2.SVM()

svm.load("training.xml")
z = np.array([1903.0,215.898617377,7689.5,366.350285411,40.0,24.6274166107,1444,459.0,1113.255,255,255,255,0,0,0],dtype = np.float32)

print svm.predict(z)
