import cv2
import numpy as np


def process_line(thresh,output):	
	# assign a rectangle kernel size	1 vertical and the other will be horizontal
	kernel = np.ones((1,5), np.uint8)
	kernel2 = np.ones((2,4), np.uint8)	
	# use closing morph operation but fewer iterations than the letter then erode to narrow the image	
	temp_img = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel2,iterations=2)
	#temp_img = cv2.erode(thresh,kernel,iterations=2)	
	line_img = cv2.dilate(temp_img,kernel,iterations=5)
	
	(contours, _) = cv2.findContours(line_img.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(output,(x-1,y-5),(x+w,y+h),(0,255,0),1)

	return output	

img = cv2.imread('test_rasture.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, imgBinary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


kernel = np.ones((1,5), np.uint8)
kernel2 = np.ones((2,4), np.uint8)

temp_img = cv2.morphologyEx(imgBinary,cv2.MORPH_CLOSE,kernel2,iterations=2)
	#temp_img = cv2.erode(thresh,kernel,iterations=2)	
line_img = cv2.dilate(temp_img,kernel,iterations=5)
cv2.imshow('image',line_img)
cv2.waitKey(0)




