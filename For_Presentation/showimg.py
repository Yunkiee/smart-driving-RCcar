import numpy as np
import cv2
frame = cv2.imread('/home/capstone/Desktop/Driving/Input_data_Image/test.png')
frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow('result',frame)
cv2.waitKey(0)
