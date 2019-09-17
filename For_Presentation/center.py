import cv2
import numpy as np

img = cv2.imread('/home/capstone/Desktop/Driving/Input_data_Image/main.jpg')  #load an image of a single battery
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to grayscale
ret, thresh = cv2.threshold(img_gs, 250, 1, cv2.THRESH_BINARY) #binary threshold
thresh = 1 - thresh  #invert: 1 for the battery, 0 for the background

h, w = thresh.shape

mat = [[col, row] for col in range(w) for row in range(h) if thresh[row, col] != 0]
mat = np.array(mat).astype(np.float32)#have to convert type for PCA

m, e = cv2.PCACompute(mat, mean = np.array([]))

center = tuple(m[0])
endpoint1 = tuple(m[0] + e[0]*100)
endpoint2 = tuple(m[0] + e[1]*50)

red_color = (0, 0, 255)
cv2.circle(img, center, 5, red_color)
cv2.line(img, center, endpoint1, red_color)
cv2.line(img, center, endpoint2, red_color)
cv2.imwrite('/home/capstone/Desktop/Driving/Input_data_Image/out.jpg', img)
