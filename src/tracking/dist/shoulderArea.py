#!/usr/bin/env python2

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('141dist.png',0)

contours, hierarchy = cv2.findContours(img,1,2)
cnt = contours[0]

img[:] = (255)
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img,ellipse,(0,0,0),2)
contours1,hierarchy1 = cv2.findContours(img, 1, 2)
cnt1 = contours1[0]
area = cv2.contourArea(cnt1)
perimeter = cv2.arcLength(cnt1,True)
plt.imshow(img), plt.show()
print area
print perimeter