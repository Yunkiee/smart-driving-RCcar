import numpy as np
import cv2

def showVideo():
	try: 
		#cap = cv2.VideoCapture('/home/capstone/Desktop/Driving/Input_data_mp4/Load3.mp4')
		cap=cv2.VideoCapture(0) 
	except:
		return

	cap.set(3, 640)
	cap.set(4, 640)

	while True:
		ret, frame = cap.read()
		
		if not ret:
			break

		gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cv2.imshow('/home/capstone/Desktop/Driving',frame)

		k=cv2.waitKey(1) & 0xFF
		if k==27:
			break

	cap.release()
	cv2.destroyAllWindows()

showVideo()
