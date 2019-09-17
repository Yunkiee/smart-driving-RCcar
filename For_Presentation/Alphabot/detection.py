import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
import os, sys
import RPi.GPIO as GPIO
import time
from AlphaBot import AlphaBot

Ab = AlphaBot()

fit_result, l_fit_result, r_fit_result, L_lane,R_lane = [], [], [], [], []
Image_xsize = 640
Image_ysize = 480

def grayscale(img):
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	mask = np.zeros_like(img) #빈 이미지 생성    
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
        
	cv2.fillPoly(mask, vertices, ignore_mask_color)
    
	masked_image = cv2.bitwise_and(img, mask)
	cv2.imshow('/home/capstone/Desktop/Driving/Input_data_mp4/result111',masked_image) 
	return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_circle(img,lines, color=[0, 0, 255]):
	for line in lines:
		cv2.circle(img,(line[0],line[1]), 2, color, -1)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_arr = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)

def Collect_points(lines):
	interp = lines.reshape(lines.shape[0]*2,2)
	for line in lines:
		if np.abs(line[3]-line[1]) > 5:
			tmp = np.abs(line[3]-line[1])
			a = line[0] ; b = line[1] ; c = line[2] ; d = line[3]
			slope = (line[2]-line[0])/(line[3]-line[1]) 
			for m in range(0,tmp,5):
				if slope>0:
					new_point = np.array([[int(a+m*slope),int(b+m)]])
					interp = np.concatenate((interp,new_point),axis = 0)
				elif slope<0:
					new_point = np.array([[int(a-m*slope),int(b-m)]])
					interp = np.concatenate((interp,new_point),axis = 0)                
	return interp

def get_random_samples(lines):
	one = random.choice(lines)
	two = random.choice(lines)
	if(two[0]==one[0]): # extract again if values are overlapped
		while two[0]==one[0]:
			two = random.choice(lines)
	one, two = one.reshape(1,2), two.reshape(1,2)
	three = np.concatenate((one,two),axis=1)
	three = three.squeeze()
	return three

def compute_model_parameter(line):
	m = (line[3] - line[1])/(line[2] - line[0])
	n = line[1] - m*line[0]
	a, b, c = m, -1, n
	par = np.array([a,b,c])
	return par

def compute_distance(par, point):
	return np.abs(par[0]*point[:,0]+par[1]*point[:,1]+par[2])/np.sqrt(par[0]**2+par[1]**2)

def model_verification(par, lines):
	distance = compute_distance(par,lines)
	sum_dist = distance.sum(axis=0)
	avg_dist = sum_dist/len(lines)
	return avg_dist

def draw_extrapolate_line(img, par,color=(0,0,255), thickness = 3):
	x1, y1 = int(-par[1]/par[0]*img.shape[0]-par[2]/par[0]), int(img.shape[0])
	x2, y2 = int(-par[1]/par[0]*(img.shape[0]/2+100)-par[2]/par[0]), int(img.shape[0]/2+100)
	cv2.line(img, (x1 , y1), (x2, y2), color, thickness)
	return img

def get_fitline(img, f_lines):

	rows,cols = img.shape[:2]
	output = cv2.fitLine(f_lines,cv2.DIST_L2,0, 0.01, 0.01)
	vx, vy, x, y = output[0], output[1], output[2], output[3]
	x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
	x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
	result = [x1,y1,x2,y2]

	return result

def draw_fitline(img, result_l,result_r, color=(0,255,0), thickness = 20): #Show line
	lane = np.zeros_like(img)
	cv2.line(lane, (int(result_l[0]) , int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
	cv2.line(lane, (int(result_r[0]) , int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
	final = weighted_img(lane, img, 1,0.5)  
	return final

def erase_outliers(par, lines):
	distance = compute_distance(par,lines)
	filtered_lines = lines[distance<13,:]
	return filtered_lines

def smoothing(lines, pre_frame):
	lines = np.squeeze(lines)
	avg_line = np.array([0,0,0,0])
	for ii,line in enumerate(reversed(lines)):
		if ii == pre_frame:
			break
		avg_line += line
	avg_line = avg_line / pre_frame
	
	return avg_line

def ransac_line_fitting(img, lines, min=100):
	global fit_result, l_fit_result, r_fit_result
	best_line = np.array([0,0,0])
	if(len(lines)!=0):                
		for i in range(30):           
			sample = get_random_samples(lines)
			parameter = compute_model_parameter(sample)
			cost = model_verification(parameter, lines)                        
			if cost < min: # update best_line
				min = cost
				best_line = parameter
			if min < 3: break
		filtered_lines = erase_outliers(best_line, lines)
		fit_result = get_fitline(img, filtered_lines)
	else:
		if (fit_result[3]-fit_result[1])/(fit_result[2]-fit_result[0]) < 5:
			l_fit_result = fit_result
			return l_fit_result
		else:
			r_fit_result = fit_result
			return r_fit_result

	if (fit_result[3]-fit_result[1])/(fit_result[2]-fit_result[0]) < 5:
		l_fit_result = fit_result
		return l_fit_result
	else:
		r_fit_result = fit_result
		return r_fit_result

def detect_lanes_img(img):
	height, width = img.shape[:2]
	vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32) #Start
	ROI_img = region_of_interest(img, vertices)
	blur_img = gaussian_blur(img, 3)#blur_img = gaussian_blur(ROI_img, 3)
	canny_img = canny(blur_img, 70, 210)
	vertices2 = np.array([[(52,height),(width/2-43, height/2+62), (width/2+43, height/2+62), (width-52,height)]], dtype=np.int32)
	canny_img = region_of_interest(canny_img, vertices2)
	line_arr = hough_lines(canny_img, 1, 1 * np.pi/180, 30, 10, 20) #def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): 
	
	if line_arr is None:
		print("Nothing")
		return img
	line_arr = np.squeeze(line_arr)
	slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

	line_arr = line_arr[np.abs(slope_degree)<160]
	slope_degree = slope_degree[np.abs(slope_degree)<160]
	line_arr = line_arr[np.abs(slope_degree)>95]
	slope_degree = slope_degree[np.abs(slope_degree)>95]
	L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]

	if L_lines is None and R_lines is None:
		print("Nothing")
		return img

	L_interp = Collect_points(L_lines)
	R_interp = Collect_points(R_lines)
	left_fit_line = ransac_line_fitting(img, L_interp)
	right_fit_line = ransac_line_fitting(img, R_interp)
	View_center = width/2
	if left_fit_line[2] == right_fit_line[2] : #One Line detection 차량의 크기의 절반을 중간값으로 설정하는 솔루션 필요 (2018.05.21)
		#print(left_fit_line)
		center_circle1= View_center #center_circle1 = (left_fit_line[2]*2)
		center_circle2 = (left_fit_line[3])
	else : #Two Line detection
		#print(left_fit_line)
		#print(right_fit_line)
		center_circle1 = (left_fit_line[2]+right_fit_line[2])/2 #양 x축 절반
		center_circle2 = (left_fit_line[3]) # y축 그대로
#	print(left_fit_line) #([111, 359, 129, 280])
#	print(right_fit_line) #([373, 359, 329, 280]) 
#	print(left_fit_line[0])
#	print(right_fit_line[0])
	L_lane.append(left_fit_line), R_lane.append(right_fit_line)

	if len(L_lane) > 10:
		left_fit_line = smoothing(L_lane, 10)    
	if len(R_lane) > 10:
		right_fit_line = smoothing(R_lane, 10)
	final = draw_fitline(img, left_fit_line, right_fit_line)
	cv2.circle(final, (int(center_circle1),int(center_circle2)), 10, (255,0,0),3) #두 차선의 중앙값 (이동이 커도 상관없음)
	
	#print(View_center, center_circle1)
	while 1:
		if View_center < int(center_circle1) and int(center_circle1) - View_center < 10: #차선인식의 범위가 많은 경우 그 차이만큼 움직이는 것이 아닌 최소한의 차이만 이
			View_center+=1
			Ab.right()
			print("Right", View_center ," -> ", center_circle1)
		elif View_center > int(center_circle1) and View_center - int(center_circle1) < 10:
			View_center-=1
			Ab.left()
			print("Left", View_center ," <- ", center_circle1)
		else:
			print("Straight",View_center ," = ", center_circle1)
			Ab.forward()
			break
	#print("Change", View_center, center_circle1)
	cv2.circle(final, (int(View_center),int(center_circle2)), 10, (0,0,255),3) #중심점 (이동) -> 차량의 이동
	return final

#f=open('/home/capstone/Desktop/Driving/Detection.txt', "w")
#cap = cv2.VideoCapture('/home/capstone/Desktop/Driving/Input_data_mp4/challenge.mp4')
cap=cv2.VideoCapture(0)
Image_xsize=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Video size cirfirm
Image_ysize=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(3, Image_xsize) #Image size 설정
cap.set(4, Image_ysize)
while(cap.isOpened()):
	ret, frame = cap.read()
	if frame.shape[0] !=540: # resizing for challenge video
		frame = cv2.resize(frame, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA) 
	result= detect_lanes_img(frame)
	cv2.imshow('/home/capstone/Desktop/Driving/Output_data_mp4/output4.jpg',result)
#	cv2.imshow('/home/capstone/Desktop/Driv.ing/Input_data_mp4',result)
	#print(frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
f.close
cap.release()
cv2.destroyAllWindows()


"""
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray=cv2.GaussianBlur(gray, (5, 5), 0)
	_,bin = cv2.threshold(gray,120,255,1)	
	bin = cv2.dilate(bin,None)
	bin = cv2.dilate(bin,None)
	bin = cv2.erode(bin,None)
	bin = cv2.erode(bin,None)
	bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	rc=cv2.minAreaRect(contours[0])
	box=cv2.boxPoints(rc)
	for p in box:
		pt=(int(p[0]+p[0]/2),int(p[1]))
		#print(round(p[0]/2), round(p[1]/2))
		p[0]=round(p[0]/2)
		k=round(p[0])
		k1=round(p[1])
		#print(k)
		#f.write(int(p[0]))
		cv2.circle(result,pt,5,(200,0,0),2)	
		#cv2.circle(result, (right_fit_line[0]-left_fit_line[0],left_fit_line[1]-left_fit_line[3]), 10, (255,0,0),3)
#		cv2.circle(result, (360,360), 10, (255,0,0),3)
		if 700 < round(k):
			print(" ")
		else :
			print(" ")
"""


"""
import cv2
import numpy as np

im=cv2.imread('/home/capstone/Desktop/Driving/Input_data_Image/test_image.jpg')
gray=cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray, (5, 5), 0)
_,bin = cv2.threshold(gray,120,255,1)
bin = cv2.dilate(bin,None)
bin = cv2.dilate(bin,None)
bin = cv2.erode(bin,None)
bin = cv2.erode(bin,None)
bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

rc=cv2.minAreaRect(contours[0])
box=cv2.boxPoints(rc)
for p in box:
	pt=(int(p[0]+p[0]/2),int(p[1]))
	a=(int(p[0]+p[0]/2), int(p[1]))
	print(int(p[0]/2))
	cv2.circle(im,pt,5,(200,0,0),2)
	#cv2.circle(im,aa,5,(200,0,0),2)
cv2.imshow('/home/capstone/Desktop/Driving/Output_data_mp4/output4.jpg',im)
cv2.waitKey()
"""
