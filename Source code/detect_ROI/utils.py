import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyclstm
from PIL import Image
import sys, getopt
import os
import difflib
import sys
import subprocess
import shutil
import argparse
reload(sys)

#Read parameter

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
print image.shape

#File and folder 

def create_folder_filename(filename):
	path=os.path.basename(filename)
	base_name=os.path.splitext(path)[0]
	create_folder(base_name)
	return base_name

def create_folder(fold_name):
	if not os.path.exists(fold_name):
		os.makedirs(fold_name)

def move_file_folder(filename, folder):
	filename_new=os.path.join(folder, filename)
	shutil.move(filename, filename_new)

def read_file_folder(folder_name, str_type):
	file_list = []
	st=strFolderName+str_type
	for filename in glob.glob(st): #assuming gif
	    file_list.append(filename)
	return file_list


#image processing
def crop_image(img, xmin, xmax,ymin, ymax):
	return img[ymin:ymax, xmin:xmax]


def crop_image_border(img,cropX=0, cropY=0, cropWidth=0, cropHeight=0, out_path="out.png"):
	h = np.size(img, 0)
	w = np.size(img, 1)	
	if(h-cropHeight>cropY) and (w-cropWidth>cropX):
		res=img[cropY:h-cropHeight, cropX:w-cropWidth]
	else:
		res=img[cropY:h,cropX:w]
	print str(os.getpid())
	cv2.imwrite(out_path,res)

def convert_binary_image(img):
	if (len(img.shape) >= 3):
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return imgBinary

def resize_image(img, w, h):
	res = cv2.resize(img,(w, h), interpolation = cv2.INTER_CUBIC)
	return res

def rotate_image(img):
	return 0



img= resize_image(image,1653, 2339)
print img.shape
adhe_img=crop_image(img, 940,1410,670,940)
parap_img=crop_image(img, 1090,1428, 2012, 2228)

cv2.imwrite("adhe.png",adhe_img)
cv2.imwrite("parap_img.png", parap_img)