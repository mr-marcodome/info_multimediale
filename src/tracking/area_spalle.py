#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from primesense import openni2
import cv2, math

MIN_RANGE = 120
MAX_RANGE = 2000
MIN_AREA = 5000

#***********ELIMINA RUMORE*********************
def removeBlackPixels(depth, N_ITER=5):

    #vengono realizzate delle operazioni morfologiche che distorcendo in
    #parte il depth frame, eliminano i pixel neri
    kernel = np.ones((5,5),np.uint8)
    depth_dil = cv2.dilate(depth,kernel,iterations = N_ITER)
    depth_er = cv2.erode(depth_dil,kernel,iterations = N_ITER)

    #creazione di una maschera che avrà valore 1 in corrispondenza dei pixel
    #neri, e zero in corrispondenza di quelli di diverso colore
    mask = cv2.inRange(depth.copy(), 0, 1)

    #applicazione della maschera al depth frame
    depth_er = cv2.bitwise_and(depth_er,depth_er,mask=mask)

    #il depth frame viene corretto "riempendo" i pixel neri
    depth = depth + depth_er

    return depth

#********* MASK*****************************
def extractMask(depth_array_fore,minRange=MIN_RANGE,maxRange=MAX_RANGE):

    #segmentazione della maschera
    mask = cv2.inRange(depth_array_fore, minRange, maxRange)

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #eliminazione, dal vettore contours, dei contorni che hanno area superiore a MIN_AREA (quindi quelli più significativi)
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if (area>MIN_AREA):
            contours.pop(idx)

    #eliminazione dei contorni rimanenti dalla maschera
    cv2.drawContours(mask, contours, -1, 0, -1)

    #eliminazione del rumore tramite l'operazione morfologica di apertura
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

#***********MAIN*******************
def main():

    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
    p.add_argument('--v', dest = 'video_path', action = 'store', default = '', help = 'path file *.oni')
    args = p.parse_args()

    #inizializzazione di OpenNI e apertura degli stream video
    openni2.initialize()
    dev = openni2.Device.open_file(args.video_path)
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    depth_stream.start()
    color_stream.start()

    frame_count = 0

    while (True):
        #acquisizione degli array relativi ai frame dallo stream RGB e Depth
        frame_depth = depth_stream.read_frame()
        frame_color = color_stream.read_frame()
        #conversione di tipo
        frame_depth_data = frame_depth.get_buffer_as_uint16()
        frame_color_data = frame_color.get_buffer_as_uint8()
        #conversione degli array in un formato utilizzabile da OpenCV
        depth_array = np.ndarray((frame_depth.height, frame_depth.width), dtype = np.uint16, buffer = frame_depth_data)
        color_array = np.ndarray((frame_color.height, frame_color.width, 3), dtype = np.uint8, buffer = frame_color_data)
        color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)
        frame_count += 1

        #se il frame è il primo, può essere preso come background del canale depth
        if frame_count == 1:
            depth_array_back = np.ndarray((frame_depth.height, frame_depth.width), dtype = np.uint16, buffer = frame_depth_data)
            depth_array_back = depth_array
            depth_array_back = removeBlackPixels(depth_array_back)

        depth_array = removeBlackPixels(depth_array)

        depth_array_fore = cv2.absdiff(depth_array, depth_array_back)

        #estrazione della maschera dal depth foreground
        mask = extractMask(depth_array_fore)

        #cv2.imshow("RGB", color_array)
        depth_array = depth_array/10000.
        cv2.imshow("Depth", depth_array)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    depth_stream.stop()
    color_stream.stop()
    openni2.unload()
    cv2.destroyAllWindows()


if __name__ == '__main__':
        main()