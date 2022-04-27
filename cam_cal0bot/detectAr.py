#####
# Versión mejorada
#####

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math



#---------------------------------------------------------------------------------------------------------------
#----------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#---------------------------------------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix > y <

def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
	
def rotationMatrixToEulerAngles(R):
	assert (isRotationMatrix(R))
	
	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1,0] * R[1,0])
	singular = sy < 1e-6
	
	if not singular:
		x = math.atan2(R[2,1], R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else:
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0	
		
	return np.array([x,y,z])




####
prev_frame_time = time.time()

cal_image_count = 0
frame_count = 0
####


marker_size = 66

with open('camera_cal.npy', 'rb') as f:
	camera_matrix = np.load(f)
	camera_distortion = np.load(f)
	
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)


camera_width = 640
camera_height = 480
camera_frame_rate = 40

cap.set(2, camera_width)
cap.set(4, camera_height)
cap.set(5, camera_frame_rate)




while True:
	
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)
	

	if ids is not None:
	#if ids is not None and ids[0] == 32:   MAL OSTIA ESTÁS PUTO CIEGO JODERRRR
	
		zipped = zip(ids, corners)
		print('zipped')
		print(zipped)
		ids, corners = zip(*(sorted(zipped)))
		print('amongas')
		print(zip(*(sorted(zipped))))
		
		for i in range(len(ids)):

			
			if ids[i] == [17]:
				pos = np.where(ids==[17])[0][0]
				color = 0
				print('rock')
				
			elif ids[i] == [47]:
				pos = np.where(ids==[47])[0][0]
				color = 1
				print('red')
				
				
			elif ids[i] == [36]:
				pos = np.where(ids==[36])[0][0]
				color = 2
				print('green')
				
			
			elif ids[i] == [13]:
				pos = np.where(ids==[13])[0][0]
				color = 3
				print('blue')
				


				

				
		print("ids")

		
		#print("corners")
		#print(corners)
		
		#print("frame")
		#print(frame)
		#pos = np.where(ids==[32])[0][0]
		#print(pos)
		
		
		aruco.drawDetectedMarkers(frame, corners)
		
		rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
		print('rvec_list_all')
		print(rvec_list_all)
		rvec = rvec_list_all[pos][0]
		tvec = tvec_list_all[pos][0]
		#print('rvec')
		#print(rvec)
		
		if color == 0:
			rvec_list_all_0, tvec_list_all_0, _objPoints_0 = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
			rvec_0 = rvec_list_all_0[pos][0]
			tvec_0 = tvec_list_all_0[pos][0]
			aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec_0, tvec_0, 100)
			coordenadas_roca = (tvec_0[0], tvec_0[1], tvec_0[2])
			print('coordenadas_roca')
			print(coordenadas_roca)	
				
		elif color == 1:
			rvec_list_all_1, tvec_list_all_1, _objPoints_1 = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
			rvec_1 = rvec_list_all_1[pos][0]
			tvec_1 = tvec_list_all_1[pos][0]
			aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec_1, tvec_1, 100)
			coordenadas_rojas = (tvec_1[0], tvec_1[1], tvec_1[2])
			print('coordenadas_rojas')
			print(coordenadas_rojas)		
			
		elif color == 2:	
			rvec_list_all_2, tvec_list_all_2, _objPoints_2 = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
			rvec_2 = rvec_list_all_2[pos][0]
			tvec_2 = tvec_list_all_2[pos][0]	
			aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec_2, tvec_2, 100)
			coordenadas_verdes = (tvec_2[0], tvec_2[1], tvec_2[2])
			print('coordenadas_verdes')
			print(coordenadas_verdes)
						
			
		elif color == 3:
			rvec_list_all_3, tvec_list_all_3, _objPoints_3 = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
			rvec_3 = rvec_list_all_3[pos][0]
			tvec_3 = tvec_list_all_3[pos][0]
			aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec_3, tvec_3, 100)
			coordenadas_azules = (tvec_3[0], tvec_3[1], tvec_3[2])
			print('coordenadas_azules')
			print(coordenadas_azules)	
			
		
		
		
		
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time
		cv2.putText(frame, 'FPS ' + str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 3, (100,255,0), 2, cv2.LINE_AA)
		
		
	cv2.imshow('frame', frame)
		
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'): break
	
cap.release()
cv2.destroyAllWindows()
		
