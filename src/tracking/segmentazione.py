#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from primesense import openni2
import cv2, math
import Image

MIN_RANGE=150
MAX_RANGE=2500
MIN_AREA=5000
MIN_HEIGHT=1200
N_ITER = 5

EXT = ".csv"

def distanza(p1,p2):
		
	#calcola la distanza geometrica tra due punti in un frame		
	dis = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
	return math.sqrt(dis)
	
def extractShoulderWidth(mask, maxd):
	
	#calcola la larghezza delle spalle
	#inizializzazione STAR detector
	orb = cv2.ORB()

	#ricerca dei keypoints con ORB
	kp = orb.detect(maskPropS,None)
	pn1, pn2 = [0,0]
	for punto in kp:
		for punto2 in kp:
			#calcolo la distanza dei due keypoint					
			dist = distanza(punto.pt,punto2.pt)
			if dist > maxdist:
				maxdist = dist
				pn1 = punto
				pn2 = punto2
				#compute the descriptors with ORB
				kp, des = orb.compute(maskPropS, kp)
				#draw only keypoints location,not size and orientation
				distimg = cv2.drawKeypoints(maskPropS,[pn1,pn2],color=(0,255,0), flags=0)
				os.chdir("dist")
				cv2.imwrite(str(i)+"dist.png",distimg)
				os.chdir("..")

def removeBlackPixels(depth):
	
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

def extractMask(depth_array_fore):

	#segmentazione della maschera
	mask = cv2.inRange(depth_array_fore, MIN_RANGE, MAX_RANGE)
	
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

def extractMaskPropShoulder(depth_array_fore, H):

	#segmentazione della maschera
	mask = cv2.inRange(depth_array_fore, 1300, H - 230)
	
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
	maskPropS = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	return maskPropS

def extractMaskPropHead(depth_array_fore, H):

	#segmentazione della maschera
	mask = cv2.inRange(depth_array_fore, H-230, H)
	
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
	maskPropH = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	return maskPropH

def getMaxHeight(depth, mask):

	#applicazione della maschera, così si è certi che il massimo venga
	#trovato sopra al soggetto
	masked = cv2.bitwise_and(depth,depth,mask = mask)
	_,H,_,posmax = cv2.minMaxLoc(masked)
	
	return H, posmax[0], posmax[1]

def getMinHeight(depth, mask):

	#applicazione della maschera, così si è certi che il minimo venga
	#trovato sopra al soggetto	
	masked = cv2.bitwise_and(depth,depth,mask = mask)
	masked2 = masked +65535	
	h,_,posmin,_ = cv2.minMaxLoc(masked2)
	
	return h, posmin[0], posmin[1]

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
	
	#estrazione dell'id della persona dal nome del file .oni
	VideoId=args.video_path.split(".")[0]
	#file con i punti ad altezza massima dei frame contenenti il soggetto
	tracking_file_color = open(VideoId + "_color" + EXT,"w")
	#file con i punti ad altezza massima di tutti i frame del video
	tracking_file_all = open(VideoId + "_depth" + EXT,"w")
    
	#contiene il timestamp del frame precedente
	t_prev = -2
	#contiene il timestamp del frame corrente
	t_curr = -1
	#indice incrementato ogni volta che si salvano i frame
	i = 0
	#indice che conta i frame aperti dallo stream video
	frame_count = 0
	
	#variabile gestione ultime mod
	ultimopassaggio=0
	newid=True
	contperid=0
	i=1
	HMAX = 0
	maxdist = 0
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
		
		#aggiornamento dei timestamp
		t_prev = t_curr
		t_curr = frame_color.timestamp
		if (t_curr < t_prev):
			break
		
		#se il frame è il primo, può essere preso come background del canale depth
		if frame_count == 1:
			depth_array_back = np.ndarray((frame_depth.height, frame_depth.width), dtype = np.uint16, buffer = frame_depth_data)
			depth_array_back = depth_array
			depth_array_back = removeBlackPixels(depth_array_back)

		depth_array = removeBlackPixels(depth_array)
		
		depth_array_fore = cv2.absdiff(depth_array, depth_array_back)

		#estrazione della maschera dal depth foreground
		mask = extractMask(depth_array_fore)
		H, x, y = getMaxHeight(depth_array_fore, mask)
		if H>HMAX:
			HMAX = H	
		
		if (H>MIN_HEIGHT):
			#Creazione maschera personalizzata sull'altezza della persona per calcolo larghezza spalle
			maskPropS = extractMaskPropShoulder(depth_array_fore, H)
			
			#inizializzazione STAR detector
			orb = cv2.ORB()

			#ricerca dei keypoints con ORB
			kp = orb.detect(maskPropS,None)
			pn1, pn2 = [0,0]
			for punto in kp:
				for punto2 in kp:
					#calcolo la distanza dei due keypoint					
					dist = distanza(punto.pt,punto2.pt)
					if dist > maxdist:
						maxdist = dist
						pn1 = punto
						pn2 = punto2
						#compute the descriptors with ORB
						kp, des = orb.compute(maskPropS, kp)
						#draw only keypoints location,not size and orientation
						distimg = cv2.drawKeypoints(maskPropS,[pn1,pn2],color=(0,255,0), flags=0)
						os.chdir("dist")
						cv2.imwrite(str(i)+"dist.png",distimg)
						os.chdir("..")
		
		if (x > 200 and x < 400):
			#Creazione maschera personalizzata sull'altezza della persona per calcolo area Testa
			maskPropH = extractMaskPropHead(depth_array_fore, H)
			os.chdir("head")
			cv2.imwrite(str(i)+"head.png",maskPropH)
			os.chdir("..")
		
		#Calcolo altezza minima
		h, u, v =getMinHeight(depth_array_fore, mask)
		#se il punto ad altezza massima nel frame depth è maggiore della soglia, si salvano le immagini della maschera
		if (H>MIN_HEIGHT):
			os.chdir("blob")
			cv2.imwrite(str(i)+"blob.png",mask)
			os.chdir("..")
			i=i+1

		#se il punto ad altezza massima nel frame depth è maggiore della soglia, si salvano le immagini
		if (H>MIN_HEIGHT):
			#gestione più persone
			if (newid==True):
				contperid+=1
				newid=False
				HMAX = 0
				maxdist = 0
			
			
			cv2.circle(depth_array,tuple((x,y)), 5, 65536, thickness=1)
			
			line_to_write = VideoId+";"+  str("{:03d}".format(contperid)) +";"+str(frame_count)+";"+str(frame_depth.timestamp)+";"+str(h)+";"+str(H)+";"+str(u)+";"+str(v)+";"+str(x)+";"+str(y)+";"+str(HMAX)+";"+str(maxdist)+"\n"
			print line_to_write
			tracking_file_all.write(line_to_write)
			line_to_write_color = VideoId+";"+ str("{:03d}".format(contperid))+";"+str(frame_count)+";"+str(frame_color.timestamp)+"\n"
			tracking_file_color.write(line_to_write_color)
			
			cv2.circle(depth_array,tuple((x,y)), 5, 65536, thickness=7)
			cv2.circle(depth_array,tuple((u,v)), 5, 65536, thickness=7)
			ultimopassaggio=frame_count+3 #3 indica quanti frame devono passare dopo il passaggio dell'ultima persona
			
		else:
			line_to_write =  VideoId+";"+ "NULL"+";"+ str(frame_count)+";"+str(frame_depth.timestamp)+";"+ "NULL"+";"+ "NULL"+";"+ "NULL"+"\n"
			print line_to_write
			

			tracking_file_all.write(line_to_write)
			line_to_write_color = VideoId+";"+ "NULL" +";"+str(frame_count)+";"+str(frame_color.timestamp)+"\n"
			tracking_file_color.write(line_to_write_color)
			#gestione multipersone
			if (frame_count>ultimopassaggio):
				newid=True;
		
		#cv2.imshow("RGB", color_array)
		depth_array = depth_array/10000.		
		cv2.imshow("Depth", depth_array)
			
		ch = 0xFF & cv2.waitKey(1)
		if ch == 27:
			break	
		
	tracking_file_color.close()
	tracking_file_all.close()
	depth_stream.stop()
	color_stream.stop()
	openni2.unload()
	cv2.destroyAllWindows()

if __name__ == '__main__':
        main()

