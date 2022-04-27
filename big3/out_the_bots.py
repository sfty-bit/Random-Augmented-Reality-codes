import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math


####
prev_frame_time = time.time()

cal_image_count = 0
frame_count = 0

noname = 0
marker_size = 66

YELLOW = 867
PURPLE = 999

ROBOT = YELLOW
RobotYellow = False
RobotPurple = False
colorRock = False
colorRed = False
colorGreen = False
colorBlue = False
center = False
list = []
####

distancias =[]

with open('camera_cal.npy', 'rb') as f:
	camera_matrix = np.load(f)
	camera_distortion = np.load(f)
	
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)


while True:
	
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)
	
	aruco.drawDetectedMarkers(frame, corners)

	if ids is not None and [47, 13, 36, 17, 42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in ids:

		rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
		
		
	# VECTORES DE LOS IDS
		for i, x in enumerate(ids):
	# ><
			print(ids)
			if x == 0 :
				tvec_robotP = tvec_list_all[i][0]
				RobotPurple = True
				
				
			elif x == 2:
				tvec_robotY = tvec_list_all[i][0]
				RobotYellow = True	
				
			elif x == 7:
				tvec_rock = tvec_list_all[i][0]
				colorRock = True
			
					
			elif x == 47:
				tvec_red = tvec_list_all[i][0]
				colorRed = True
				
			elif x == 36:
				print(i)
				tvec_green = tvec_list_all[i][0]
				colorGreen = True
				
				for i,x in enumerate(tvec_green):
					list.append(x)
					print(list)
				
			elif x == 13:
				tvec_blue = tvec_list_all[i][0]
				colorBlue = True
				
			elif x == 42:
				tvec_center = tvec_list_all[i][0]
				center = True
		
	# DISTANCIAS CON RESPECTO AL EQUIPO MORADO
		if ROBOT == PURPLE:
			if RobotPurple and colorRock == True:
				dist_rock = np.linalg.norm(tvec_robotP - tvec_rock)	
				if dist_rock < dist_rock1:
					dist_rock1 == dist_rock
					print(dist_rock1)	
				#print('dist_rock')
				#print(dist_rock)
				#distancias.append(dist_rock)
				cv2.putText(frame, 'dist_rock: ' + str(int(dist_rock)), (10,450), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)			
					
			if RobotPurple and colorRed == True:
				dist_red = np.linalg.norm(tvec_robotP - tvec_red)
				cv2.putText(frame, 'dist_red: ' + str(int(dist_red)), (10,300), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)	
				distancias.append(dist_red)
				
			if RobotPurple and colorGreen == True:
				dist_green = np.linalg.norm(tvec_robotP - tvec_green)
				cv2.putText(frame, 'dist_green: ' + str(int(dist_green)), (10,350), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)				
				distancias.append(dist_green)
			if RobotPurple and colorBlue == True:
				dist_blue = np.linalg.norm(tvec_robotP - tvec_blue)
				cv2.putText(frame, 'dist_blue: ' + str(int(dist_blue)), (10,400), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)
				distancias.append(dist_blue)
				
			if RobotPurple and center == True:
				dist_center = np.linalg.norm(tvec_robotP - tvec_center)
				cv2.putText(frame, 'center: ' + str(int(dist_center)), (200,40), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)
				
	# DISTANCIAS CON RESPECTO AL EQUIPO AMARILLO
		if ROBOT == YELLOW:
			if RobotYellow and colorRock == True:
				dist_rock = np.linalg.norm(tvec_robotY - tvec_rock)
				cv2.putText(frame, 'dist_rock: ' + str(int(dist_rock)), (10,450), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)	
				distancias.append(dist_rock)		
					
			if RobotYellow and colorRed == True:
				dist_red = np.linalg.norm(tvec_robotY - tvec_red)
				cv2.putText(frame, 'dist_red: ' + str(int(dist_red)), (10,300), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)	
				distancias.append(dist_red)
							
			if RobotYellow and colorGreen == True:
				dist_green = np.linalg.norm(tvec_robotY - tvec_green)
				cv2.putText(frame, 'dist_green: ' + str(int(dist_green)), (10,350), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)				
				distancias.append(dist_green)
			if RobotYellow and colorBlue == True:
				dist_blue = np.linalg.norm(tvec_robotY - tvec_blue)
				cv2.putText(frame, 'dist_blue: ' + str(int(dist_blue)), (10,400), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)
				distancias.append(dist_blue)
				
			if RobotYellow and center == True:
				dist_center = np.linalg.norm(tvec_robotP - tvec_center)
				cv2.putText(frame, 'dist_center: ' + str(int(dist_center)), (200,40), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)
		print('distancias')
		print(distancias)
		
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time
		cv2.putText(frame, 'FPS ' + str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 3, (100,255,0), 2, cv2.LINE_AA)
		[ 1, 4, 6, 7, 15] 
		
	cv2.imshow('frame', frame)
		
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'): break
	
cap.release()
cv2.destroyAllWindows()
	
	
	#dict = {'Inglaterra': 'londres', 'España':'Madrid', 'elbicho':7}
	
#https://stackoverflow.com/questions/1060090/changing-variable-names-with-python-for-loops
# ><
