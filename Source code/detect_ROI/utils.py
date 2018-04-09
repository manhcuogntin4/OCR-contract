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
import glob
import re
reload(sys)

# os call 

def convert_pdf_png(filename):
	rc=subprocess.check_call(["./convert_pdf_png.sh", filename])


#File and folder 

def count_page_pdf(filename):
	rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)  
	data = file(filename,"rb").read()  
	return len(rxcountpages.findall(data)) 

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
	st=folder_name+str_type
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

def write_crop_img(filename,str_ext,reg_string):
	if check_file_name(filename, reg_string):
			print "here"
			img=cv2.imread(filename,0)
			parap_img=crop_image(img, 1090,1428, 2038, 2228)
			base_name=get_base_name(filename)
			dirname=get_path_name(filename)
			parap_img_path=set_file_name(dirname, base_name, "_parap.png")
			print parap_img_path
			cv2.imwrite(parap_img_path, parap_img)

def rotate_image(img):
	return 0

# String processing

def check_file_name(filename, reg_string):
	regexex=reg_string
	if re.match(regexex, filename) is not None: 
		return True
	else:
		return False

# Path processing

def get_base_name(filename):
	return os.path.splitext(os.path.basename(filename))[0]
def get_path_name(filename):
	return os.path.dirname(filename)

def set_file_name(dirname, basename, extname):
	return os.path.join(dirname, basename+ extname)




if __name__ == '__main__':
	#filename = os.path.join(this_dir, 'demo', 'prenom0.png')
	#Read parameter construct the argument parser and parse the arguments
	#Unit test
	''' Test 1 crop image
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "Path to the image to be scanned")
	args = vars(ap.parse_args())
	image = cv2.imread(args["image"])
	print image.shape
	img= resize_image(image,1653, 2339)
	print img.shape
	adhe_img=crop_image(img, 940,1410,670,940)
	parap_img=crop_image(img, 1090,1428, 2012, 2228)

	cv2.imwrite("adhe.png",adhe_img)
	cv2.imwrite("parap_img.png", parap_img)
	'''
	''' Test 2 check file name
	filename="ABC-3.png"
	reg_string="(.)*(-)[0-5].png"
	file_list=read_file_folder("./", "*.png")
	print file_list
	for file in file_list:
		if check_file_name(file, reg_string):
			print "here"
			img=cv2.imread(file,0)
			parap_img=crop_image(img, 1090,1428, 2038, 2228)
			base_name=get_base_name(file)
			dirname=get_path_name(file)
			parap_img_path=set_file_name(dirname, base_name, "_parap.png")
			print parap_img_path
			cv2.imwrite(parap_img_path, parap_img)
	print check_file_name(filename, reg_string)
	''' 
	''' Test 3 :  create_folder and covert_file
	file_list=read_file_folder("./", "*.pdf")
	for file in file_list:
		folder_path=create_folder_filename(file)
		move_file_folder(file, folder_path)
		file_path=os.path.join(folder_path,file)
		convert_pdf_png(file_path)
		img_list=read_file_folder(folder_path+"/", "*.png")
		print img_list
		for img_path in img_list:
			reg_string="(.)*(-)[0-5].png"
			write_crop_img(img_path, "_parap.png", reg_string)
	'''
	
	filename='CA5.pdf'
	print count_page_pdf(filename)

