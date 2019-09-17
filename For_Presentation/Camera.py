import numpy as np
import cv2
#cap = cv2.VideoCapture('/home/capstone/Desktop/Driving/Input_data_Image/test.png')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('/home/capstone/Desktop/Driving/Operation/Essential/output.mp4', fourcc, 10.0, (640,480))
#ffmpeg -i TheGoodTheBadAndTheUgly.mp4 -vf  "setpts=4*PTS" DownTheGoodTheBadAndTheUgly.mp4
while True : 
	ret, frame = cap.read()
#	frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA) 
#	cv2.imshow('output',frame)
	cv2.imshow('/home/capstone/Desktop/Driving/Operation/Essential',frame)
	output.write(frame)
	k=cv2.waitKey(1) & 0xFF
	if k==27:
		break

cap.release()
output.release()
cv2.destroyAllWindows()








