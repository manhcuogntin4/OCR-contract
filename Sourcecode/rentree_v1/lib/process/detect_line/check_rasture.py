import cv2
import numpy as np
import argparse



def convert_binary_image(img):
	if (len(img.shape) >= 3):
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return imgBinary

def count_pixel_lines(img):
	imgBinary = convert_binary_image(img)
	kernel = np.ones((1,5), np.uint8)
	kernel2 = np.ones((2,4), np.uint8)

	temp_img = cv2.morphologyEx(imgBinary,cv2.MORPH_CLOSE,kernel2,iterations=2)
		#temp_img = cv2.erode(thresh,kernel,iterations=2)	
	line_img = cv2.dilate(temp_img,kernel,iterations=5)
	#cv2.imshow('image',line_img)
	idx=line_img<1
	count= np.sum(idx)
	#cv2.imwrite("out.png",line_img)
	print count
	return count

def is_rasture(img, threshold=500):
	if count_pixel_lines(img)>threshold:
		return True
	else:
		return False
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "Path to the image to be scanned")
	args = vars(ap.parse_args())
	img = cv2.imread(args["image"])
	if is_rasture(img, 1000):
		print "Rasture"
	else:
		print "Not Rasture"



