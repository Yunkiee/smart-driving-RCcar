import cv2 
import numpy as np

def grayscale(img): 
	return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold): 
	return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size): 
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): 
	mask = np.zeros_like(img)
	if len(img.shape) > 2: 
		color = color3
	else: 
		color = color1

	cv2.fillPoly(mask, vertices, color)
	ROI_image = cv2.bitwise_and(img, mask)
	cv2.imshow('/home/capstone/Desktop/Driving/Input_data_mp4/result111',ROI_image) 
	return ROI_image

def draw_lines(img, lines, color=[0, 255, 0], thickness=10):
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(img, (x1, y1), (x2, y2), color, thickness)
			print(x1,y1,x2,y2)		
#hough_img = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20) 
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): 
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	print(img.shape[0])
	draw_lines(line_img, lines)
	return line_img

def weighted_img(img, initial_img, α=1, β=1., λ=0.):
	return cv2.addWeighted(initial_img, α, img, β, λ)


cap = cv2.VideoCapture('/home/capstone/Desktop/Driving/Input_data_mp4/Loadview.mp4') # 동영상 불러오기
#cap=cv2.VideoCapture(0)
cap.set(3, 720) #최적의 영상크기 찾음
cap.set(4, 720)
#image.shape[0] = x축 [0] = y축
while(cap.isOpened()):	
	ret, image = cap.read()
	#output.write(image)
	height, width = image.shape[:2]
	print(image.shape[0])
	gray_img = grayscale(image)
	blur_img = gaussian_blur(gray_img, 3) 	
	canny_img = canny(blur_img, 70, 210) 
	vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
	ROI_img = region_of_interest(canny_img, vertices) 
	hough_img = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 1000) 
	result = weighted_img(hough_img, image)
	gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
		print(round(p[0]/2), round(p[1]/2))
		p[0]=round(p[0]/2)
		k=round(p[0])
		k1=round(p[1])
		#print(k)
		#f.write(int(p[0]))
		cv2.circle(result,pt,5,(200,0,0),2)	
		cv2.circle(result, (360,360), 10, (255,0,0),3)
		if 700 < round(k):
			print("right")
		else :
			print("left")
	cv2.imshow('/home/capstone/Desktop/Driving/Input_data_mp4/result',result) 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
#output.release()


"""
image=cv2.imread('/home/capstone/Desktop/Driving/Input_data_image/slope.jpg')
height, width = image.shape[:2] # 이미지 높이, 너비
gray_img = grayscale(image) # 흑백이미지로 변환
blur_img = gaussian_blur(gray_img, 3) # Blur 효과
canny_img = canny(blur_img, 70, 210) # Canny edge 알고리즘
vertices = np.array([[(50,height),(width/2-45, height/2+60), (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices) # ROI 설정
line_arr = hough_lines(ROI_IMG, 1, 1*np.pi/180, 30, 10, 20) #허프 변환
line_arr = np.squeeze(line_arr)
slope_degree=(np.arctan2(line_arr[:,1]-line_arr[:,3], line_arr[:,0] - line-arr[:,2]) * 180) /np.pi

# 수평 기울기 제한
line_arr = line_arr[np.abs(slope_degree)<160]
slope_degree = slope_degree[np.abs(slope_degree)<160]
# 수직 기울기 제한
line_arr = line_arr[np.abs(slope_degree)>95]
slope_degree = slope_degree[np.abs(slope_degree)>95]
# 필터링된 직선 버리기
L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
L_lines, R_lines = L_lines[:,None], R_lines[:,None]
# 직선 그리기
draw_lines(temp, L_lines)
draw_lines(temp, R_lines)

result = weighted_img(temp, image) # 원본 이미지에 검출된 선 overlap
cv2.imshow('result',result) # 결과 이미지 출력
cv2.waitKey(0)
"""



