#!/usr/bin/env python2

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('25dist.png',0)
contours, hierarchy = cv2.findContours(img,1,2)

i=0
for cnt in contours:
    hull = cv2.convexHull(cnt,returnPoints=False)
    defects = cv2.convexityDefects(cnt,hull)
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(img,start,end,[255,255,255],2)
            cv2.circle(img,far,5,[255,255,255],-1)

cv2.imshow("img",img)

cv2.waitKey(0)

cv2.destroyAllWindows()
print "hello"