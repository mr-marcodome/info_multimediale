#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
from primesense import openni2
import cv2, math

MIN_RANGE = 150
MAX_RANGE = 2500
MIN_AREA = 1500
MIN_AREAH = 500
MIN_HEIGHT = 1200
N_ITER = 5
DIR1 = "dist"
DIR2 = "head"
DIR3 = "shoulderArea"

EXT = ".csv"

def distanza(p1,p2):
		
	#calcola la distanza geometrica tra due punti in un frame
	#usata per calcolare la larghezza delle spalle		
	dis = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
	return math.sqrt(dis)

def getMaxValIndex(npVec):
	if len(npVec) is not 0:
		maxN=0
		maxIndex=0
		for indice,elem in enumerate(npVec):
			if elem > maxN:
				maxN = elem
				maxIndex = indice
		return maxIndex

	else:
		raise 'Error: The Array must not be Empty!'

def calculateShoulderWidth(maskPropS, maxd, i, pn1, pn2):
	
	#calcola la larghezza delle spalle
	mdist = [0]
	#inizializzazione STAR detector
	orb = cv2.ORB()
	#ricerca dei keypoints con ORB
	kp = orb.detect(maskPropS,None)
	for punto in kp:
		for punto2 in kp:
			#calcolo la distanza dei due keypoint					
			dist = distanza(punto.pt,punto2.pt)
			mdist.append(dist)
			if dist > maxd:
				maxd = dist
				pn1 = punto
				pn2 = punto2
				#Crea un immagine con i keypoint delle spalle e la salva
				distimg = cv2.drawKeypoints(maskPropS,[pn1,pn2],color=(0,255,0), flags=0)
				os.chdir(DIR1)
				cv2.imwrite(str(i)+"dist.png", distimg)
				os.chdir("..")
			pn1 = punto
			pn2 = punto2

	return maxd, max(mdist), (int(pn1.pt[0]), int(pn1.pt[1])), (int(pn2.pt[0]), int(pn2.pt[1]))
	
def calculateShoulderAreaPerimeter(mask, i):
	
	#Calcolo dei contorni della maschera ad altezza spalle
	contours, hierarchy = cv2.findContours(mask,1,2)
	area = 0
	perimeter = 0
	try:
		cnt = contours[0]
		if len(cnt) >= 5:
			#L'immagine della maschera viene riempita solo con pixel neri
			mask[:] = (255)
			#Approssimazione dei contorni della maschera
			ellipse = cv2.fitEllipse(cnt)
			cv2.ellipse(mask,ellipse,(0,0,0),2)
			contours1,hierarchy1 = cv2.findContours(mask, 1, 2)
			cnt1 = contours1[0]
			#Calcolo area e perimetro
			area = cv2.contourArea(cnt1)
			perimeter = cv2.arcLength(cnt1,True)
			#os.chdir(DIR3)
			#cv2.imwrite(str(i)+"sarea.png",mask)
			#os.chdir("..")
		
		return area, perimeter
		
	except IndexError:
		return area, perimeter

def calculateHeadArea(mask):
	#Calcola Area e Circonferenza della testa
	contours,hierarchy = cv2.findContours(mask, 1, 2)
	try:
		cnt = contours[0]
		return cv2.contourArea(cnt)
	except IndexError:
		return 0

def calculateHeadPerimeter(mask):
	#Calcola Area e Circonferenza della testa
	contours,hierarchy = cv2.findContours(mask, 1, 2)
	try:
		cnt = contours[0]
		return cv2.arcLength(cnt,True)
	except IndexError:
		return 0

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
	mask = cv2.inRange(depth_array_fore, H - 600, H - 280)
	
	contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#eliminazione, dal vettore contours, dei contorni che hanno area superiore a MIN_AREA (quindi quelli più significativi)
	for idx, cnt in enumerate(contours):
		area = cv2.contourArea(cnt)		
		if (area>MIN_AREA):
			contours.pop(idx)	
		
	#eliminazione dei contorni rimanenti dalla maschera
	cv2.drawContours(mask, contours, -1, 0, -1)	
	
	#eliminazione del rumore tramite l'operazione morfologica di apertura
	kernel = np.ones((4,4),np.uint8)
	maskPropS = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	maskPropS = cv2.dilate(maskPropS,kernel,iterations = 20)
	maskPropS = cv2.erode(maskPropS,kernel,iterations = 20)
	
	return maskPropS

def extractMaskPropHead(depth_array_fore, H):

	#segmentazione della maschera
	mask = cv2.inRange(depth_array_fore, H- 200, H)
	
	contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#eliminazione, dal vettore contours, dei contorni che hanno area superiore a MIN_AREA (quindi quelli più significativi)
	for idx, cnt in enumerate(contours):
		area = cv2.contourArea(cnt)		
		if (area>MIN_AREAH):
			contours.pop(idx)	
		
	#eliminazione dei contorni rimanenti dalla maschera
	cv2.drawContours(mask, contours, -1, 0, -1)	
	
	#eliminazione del rumore tramite l'operazione morfologica di apertura
	kernel = np.ones((4,4),np.uint8)
	maskPropH = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	maskPropH = cv2.dilate(maskPropH,kernel,iterations = 10)
	maskPropS = cv2.erode(maskPropH,kernel,iterations = 40)
	
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
	mask = cv2.bitwise_and(depth,depth,mask = mask)
	mask += 65535
	h,_,posmin,_ = cv2.minMaxLoc(mask)
	
	return h, posmin[0], posmin[1]

def calculateHeightDist(x, y, u, v):
	
	#Calcola la distanza del punto di minima altezza attuale da quello precedente
	p1 = (x,y)
	p2 = (u,v)
	
	return distanza(p1,p2)
	
def main():

	p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="")
	p.add_argument('--v', dest = 'video_path', action = 'store', default = '', help = 'path file *.oni')
	args = p.parse_args()
	
	#creazione delle directory in cui salvare le immagini, se non sono già presenti
	if not os.path.isdir(DIR1):
		os.mkdir(DIR1)
	
	if not os.path.isdir(DIR2):
		os.mkdir(DIR2)
		
	if not os.path.isdir(DIR3):
		os.mkdir(DIR3)
	
	#inizializzazione di OpenNI e apertura degli stream video	
	openni2.initialize()
	dev = openni2.Device.open_file(args.video_path)
	depth_stream = dev.create_depth_stream()
	color_stream = dev.create_color_stream()
	depth_stream.start()
	color_stream.start()
	
	#estrazione dell'id della persona dal nome del file .oni
	VideoId=args.video_path.split("/")[-1].split(".")[-2]
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
	HMAX = 0
	maxdist = 0
	shoulderH = 0
	shoulderd = 0
	hpelvis = 0
	a = 0
	b = 0
	j = 0
	pn1 = cv2.KeyPoint()
	pn2 = cv2.KeyPoint()
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
		colorMask = depth_array_back
		mask_foreground = depth_array_back
		#estrazione della maschera dal depth foreground
		mask = extractMask(depth_array_fore)
		H, x, y = getMaxHeight(depth_array_fore, mask)
		Hi = H
		if (y > 0):
			H = H - (y - 100)
		if (y > 300):
			H -= y*(0.21)
		print H, y
		if Hi>HMAX:
			HMAX = Hi	
		if (Hi>MIN_HEIGHT):
			
			#Stima altezza del bacino
			hpelvis = H*0.585
			#Calcolo dell'altezza minima
			h, u, v = getMinHeight(depth_array_fore, mask)
			#Calcolo distanza tra altezze minime
			#minDist = calculateHeightDist(a, b, u, v)
			#Variabili di swap per conservare le coordinate del frame precedente
			a = u
			b = v
			#Creazione maschera personalizzata sull'altezza della persona per calcolo larghezza spalle
			maskPropS = extractMaskPropShoulder(depth_array_fore, Hi)
			os.chdir(DIR3)
			cv2.imwrite(str(i)+"sarea.png",maskPropS)
			os.chdir("..")
			#Calcolo larghezza spalle
			maxdist, shoulderd, p1, p2 = calculateShoulderWidth(maskPropS, maxdist, i, pn1, pn2)
			#Calcolo altezza spalle nei punti da cui si è calcolata la larghezza spalle
			shoulderH = (depth_array_fore[p1[1], p1[0]] + depth_array_fore[p2[1], p2[0]])/2
			#Calcolo altezza testa spalle
			headShoulder = Hi-shoulderH
			#Calcolo area e perimetro spalle
			sarea, pshoulder = calculateShoulderAreaPerimeter(maskPropS, i)
		
			# Estrazione Colore Dei Vestiti in RGB
			mask_foreground = extractMaskPropShoulder(depth_array_fore, Hi)
			kernel_erode = np.ones((5,5),np.uint8)
			mask_foreground = cv2.erode(mask_foreground,kernel_erode,iterations=5)

			temp_mask = cv2.cvtColor(mask_foreground,cv2.COLOR_GRAY2RGB)
			colorMask = cv2.bitwise_and(color_array,temp_mask)

			#calcolo istogramma
			if Hi > MIN_HEIGHT:
				bChannel = cv2.calcHist([colorMask],[0],mask_foreground,[256],[0,256])
				gChannel = cv2.calcHist([colorMask],[1],mask_foreground,[256],[0,256])
				rChannel = cv2.calcHist([colorMask],[2],mask_foreground,[256],[0,256])

				colorePersona = (getMaxValIndex(rChannel),getMaxValIndex(gChannel),getMaxValIndex(bChannel))

		# Estrazione Colore Dei Capelli in RGB
			mask_foreground = extractMaskPropHead(depth_array_fore, Hi)
			mask_foreground = cv2.erode(mask_foreground,kernel_erode,iterations=5)

			temp_mask = cv2.cvtColor(mask_foreground,cv2.COLOR_GRAY2RGB)
			colorMask = cv2.bitwise_and(color_array,temp_mask)

			#calcolo istogramma
			if Hi > MIN_HEIGHT:
				bChannel = cv2.calcHist([colorMask],[0],mask_foreground,[256],[0,256])
				gChannel = cv2.calcHist([colorMask],[1],mask_foreground,[256],[0,256])
				rChannel = cv2.calcHist([colorMask],[2],mask_foreground,[256],[0,256])

				coloreCapelli = (getMaxValIndex(rChannel),getMaxValIndex(gChannel),getMaxValIndex(bChannel))

		#Serve per evitare un errato calcolo dell'area della testa in quanto ai bordi del frame 
		#la dimensione della maschera tende ad aumentare di molto
		if (x > 100 and x < 540):
			#Creazione maschera personalizzata sull'altezza della persona per calcolo area Testa
			maskPropH = extractMaskPropHead(depth_array_fore, Hi)
			os.chdir(DIR2)
			cv2.imwrite(str(i)+"head.png",maskPropH)
			os.chdir("..")
			harea= calculateHeadArea(maskPropH)
			phead = calculateHeadPerimeter(maskPropH)

		#se il punto ad altezza massima nel frame depth è maggiore della soglia, si salvano le immagini
		if (Hi>MIN_HEIGHT):
			#gestione più persone
			if (newid==True):
				contperid+=1
				newid=False
				
			if (x > 100 and x < 540 and (x + y) > 150 ):
				j += 1
				cv2.circle(depth_array,tuple((x,y)), 5, 65536, thickness=1)
				line_to_write = str("{:03d}".format(contperid)) +";"+str(H)+";"+str(shoulderd)+";"+str(harea)+";"+str(phead)+";"+str(sarea)+";"+str(pshoulder)+";"+str(hpelvis)+";"+str(shoulderH)+";"+str(headShoulder)+";"+ str(colorePersona[0])+";"+str(colorePersona[1])+";"+str(colorePersona[2])+";"+str(coloreCapelli[0])+";"+str(coloreCapelli[1])+";"+str(coloreCapelli[2])+"\n"         
				print line_to_write
				tracking_file_all.write(line_to_write)
				line_to_write_color = VideoId+";"+ str("{:03d}".format(contperid))+";"+str(frame_count)+";"+str(frame_color.timestamp)+"\n"
				tracking_file_color.write(line_to_write_color)

				if j > 60:
					exit()
					
			cv2.circle(depth_array,tuple((x,y)), 5, 65536, thickness=7)
			cv2.circle(depth_array,tuple((u,v)), 5, 65536, thickness=7)
			cv2.circle(depth_array,tuple((320,100)), 5, 65536, thickness=5)
			cv2.circle(depth_array,tuple((320,280)), 5, 65536, thickness=5)
			cv2.circle(depth_array,tuple((100,100)), 5, 65536, thickness=5)
			cv2.circle(depth_array,tuple((100,280)), 5, 65536, thickness=5)
			cv2.circle(depth_array,tuple((540,100)), 5, 65536, thickness=5)
			cv2.circle(depth_array,tuple((540,280)), 5, 65536, thickness=5)
			
			ultimopassaggio=frame_count+3 #3 indica quanti frame devono passare dopo il passaggio dell'ultima persona
			
		else:
			line_to_write =  VideoId+";"+ "NULL"+";"+ str(frame_count)+";"+str(frame_depth.timestamp)+";"+ "NULL"+";"+ "NULL"+";"+ "NULL"+"\n"
			print line_to_write
			

			#tracking_file_all.write(line_to_write)
			#line_to_write_color = VideoId+";"+ "NULL" +";"+str(frame_count)+";"+str(frame_color.timestamp)+"\n"
			#tracking_file_color.write(line_to_write_color)
			#gestione multipersone
			if (frame_count>ultimopassaggio):
				newid=True
				HMAX = 0
				maxdist = 0
				harea = 0
				phead = 0
				sarea = 0
				pshoulder = 0
				shoulderH = 0
				headShoulder = 0
				shoulderd = 0
				hpelvis = 0
				a = 0
				b = 0
				j = 0
		
		depth_array = depth_array/10000.		
		#cv2.imshow("Depth", depth_array)
		#cv2.imshow("Color", colorMask)
			
		ch = 0xFF & cv2.waitKey(1)
		if ch == 27:
			break	
		i +=1
		
	tracking_file_color.close()
	tracking_file_all.close()
	depth_stream.stop()
	color_stream.stop()
	openni2.unload()
	cv2.destroyAllWindows()

if __name__ == '__main__':
        main()

