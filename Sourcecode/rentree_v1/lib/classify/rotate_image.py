import cv2
import numpy as np
import glob
import imutils
import os
import sys
from string import Template

IMAGES_FOLDER="rotate/"

def read_image(file_path):
	img = cv2.imread(file_path,1)
	return img

def is_rotate(img):
	rows,cols,_ = img.shape
	if(rows>cols):
		return True
	else:
		return False

def rotate_image(img):
    rows,cols,_ = img.shape
    print rows, cols
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
def save_image(img, file_path):
	cv2.imwrite(file_path, img)


def readFileImages(strFolderName):
	print strFolderName
	image_list = []
	st=strFolderName+"*.png"
	for filename in glob.glob(st): #assuming gif
	    image_list.append(filename)
	return image_list

strFolderName=IMAGES_FOLDER
image_list=readFileImages(strFolderName)


def face_detection(face_cascade_path, file_path):
	face_cascade = cv2.CascadeClassifier(os.path.expanduser(face_cascade_path))
	scale_factor = 1.1
	min_neighbors = 3
	min_size = (30, 30)
	flags = cv2.CASCADE_SCALE_IMAGE
	image = cv2.imread(file_path)
	faces = face_cascade.detectMultiScale(image, scaleFactor = scale_factor, minNeighbors = min_neighbors,
	minSize = min_size, flags = flags)
	for( x, y, w, h ) in faces:
	    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
	    outfname = "%s.faces.jpg" % os.path.basename(file_path)
	    cv2.imwrite(os.path.expanduser(outfname), image)



face_cascade_path="haarcascade_frontalface_default.xml"
for file_path in image_list:
	print file_path
	img=read_image(file_path)
	if(is_rotate(img)):
		print "Here"
		#dst=rotate_image(img)
		dst=imutils.rotate_bound(img,90)
		print dst.size
		save_image(dst,file_path)
		face_detection(face_cascade_path,file_path)





	






