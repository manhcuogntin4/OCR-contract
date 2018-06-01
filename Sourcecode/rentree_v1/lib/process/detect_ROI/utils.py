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
import pytesseract
import unicodedata
from nltk.tokenize import sent_tokenize, word_tokenize
reload(sys)

# os call 
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
def convert_pdf_png(filename):
	print REPO_DIR
	file_bath=os.path.join(REPO_DIR, "convert_pdf_png.sh")
	rc=subprocess.check_call([file_bath, filename])

def convert_tif_png(filename):
	print filename, REPO_DIR
	file_bath=os.path.join(REPO_DIR, "convert_tiff_png.sh")
	rc=subprocess.check_call([file_bath, filename])

def rename_files_folder(folder_path):
	file_bath=os.path.join(REPO_DIR, "rename_file_images.sh")
	rc=subprocess.check_call([file_bath, folder_path])

def convert_file_folder(filename):
	UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
	print os.path.basename(filename)
	filename_new=os.path.join(UPLOAD_FOLDER, os.path.basename(filename))
	#shutil.move(filename, filename_new)
	shutil.copyfile(filename, filename_new)
	folder_path=create_folder_filename(filename_new)
	print filename_new, folder_path
	#folder_path=os.path.join(UPLOAD_FOLDER,folder_path)
	move_file_folder(filename_new, folder_path)
	file_path=os.path.join(folder_path,os.path.basename(filename))
	print file_path
	if file_path.endswith(".pdf"):
		print "pdf"
		convert_pdf_png(file_path)
	elif file_path.endswith(".tif"):
		print "tif"
		convert_tif_png(file_path)
	rename_files_folder(folder_path)
	img_list=read_file_folder(folder_path+"/", "*.png")
	return img_list

	


#File and folder 

def count_page_pdf(filename):
	rxcountpages = re.compile(r"/Type\s*/Page([^s]|$)", re.MULTILINE|re.DOTALL)  
	data = file(filename,"rb").read()  
	return len(rxcountpages.findall(data)) 

def create_folder_filename(filename):
	path=os.path.basename(filename)
	dirname=os.path.dirname(filename)
	base_name=os.path.splitext(path)[0]
	create_folder(os.path.join(dirname,base_name))
	return os.path.join(dirname,base_name)

def create_folder(fold_name):
	if not os.path.exists(fold_name):
		os.makedirs(fold_name)

def move_file_folder(filename, folder):
	print "folder dir", folder
	basename=os.path.basename(filename)
	print basename
	filename_new=os.path.join(folder, basename)
	print  "filname new", filename_new
	shutil.move(filename, filename_new)

def read_file_folder(folder_name, str_type):
	file_list = []
	st=folder_name+str_type
	for filename in glob.glob(st): #assuming gif
	    file_list.append(filename)
	return file_list


#image processing

def check_image(img):
	if img is not None:
		return True
	else:
		return False

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

def write_crop_img(filename,str_ext,reg_string, xmin,xmax,ymin,ymax):
	if check_file_name(filename, reg_string):
			print "here"
			img=cv2.imread(filename,0)
			img= resize_image(img,1653, 2339)
			#parap_img=crop_image(img, 1090,1428, 2038, 2228)
			parap_img=crop_image(img, xmin,xmax, ymin, ymax)
			base_name=get_base_name(filename)
			dirname=get_path_name(filename)
			parap_img_path=set_file_name(dirname, base_name, str_ext)
			print parap_img_path
			cv2.imwrite(parap_img_path, parap_img)

def rotate_image(img):
	return 0

# Text and string processing

def check_file_name(filename, reg_string):
	regexex=reg_string
	if re.match(regexex, filename) is not None: 
		return True
	else:
		return False

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def search_line(corpus, str_search, threshold=1):
	lines= sent_tokenize(corpus)
	#print len(lines)
	if len(lines)!=0:
		for line in lines:
			#print(word_tokenize(line))
			words=line.split(" ")
			#print words
			for word in words:
				d=levenshteinDistance(str_search, word)
				if d<=threshold: # threshold is number of diffirent words
					return line
	else:
		line=corpus
		words=line.split(" ")
			#print words
		for word in words:
			d=levenshteinDistance(str_search, word)
			if d<=threshold: # threshold is number of diffirent words
				return line

	return ""

def search_list(corpus, list_search, threshold=1):
	for str_search in list_search:
		if search_line(corpus, str_search, threshold):
			return True
	return False


# Remove accent
def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
        #print text
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

#Tesseract read
def read_tesseract_file(img_file,type="thresh"):
	image = cv2.imread(img_file)
	if check_image(image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		 
		# check to see if we should apply thresholding to preprocess the
		# image
		if type == "thresh":
			gray = cv2.threshold(gray, 0, 255,
				cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		 
		# make a check to see if median blurring should be done to remove
		# noise
		elif type == "blur":
			gray = cv2.medianBlur(gray, 3)
		 
		# write the grayscale image to disk as a temporary file so we can
		# apply OCR to it
		base_name=get_base_name(img_file)
		path_name=get_path_name(img_file)
		ext_name="gray"
		filename = os.path.join(path_name,ext_name+base_name+".png")
		cv2.imwrite(filename, gray)

		# load the image as a PIL/Pillow image, apply OCR, and then delete
		# the temporary file
		text = pytesseract.image_to_string(Image.open(filename))
		os.remove(filename)
		return text
	else:
		return ""

def read_tesseract_image(image,type="thresh"):
	if check_image(image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		 
		# check to see if we should apply thresholding to preprocess the
		# image
		if type == "thresh":
			gray = cv2.threshold(gray, 0, 255,
				cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		 
		# make a check to see if median blurring should be done to remove
		# noise
		elif type == "blur":
			gray = cv2.medianBlur(gray, 3)
		 
		# write the grayscale image to disk as a temporary file so we can
		# apply OCR to it
		
		filename = "gray_out.png"
		cv2.imwrite(filename, gray)

		# load the image as a PIL/Pillow image, apply OCR, and then delete
		# the temporary file
		text = pytesseract.image_to_string(Image.open(filename))
		os.remove(filename)
		return text
	else:
		return ""

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
	'''Test 4 crop all file pdf 
	
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
			write_crop_img(img_path, "_page_number.png", reg_string, 1463,1540, 2171,2230)
	'''
	
	create_folder("1")
	create_folder("2")
	search_list=["1/5","2/5","3/5","4/5","5/5","1/6","2/6","3/6","4/6","5/6","6/6"]
	img_list=read_file_folder("./", "*.png")
	for file_img in img_list:
		img_rgb=cv2.imread(file_img,1)
		img= resize_image(img_rgb,1653, 2339)
		img_crop=crop_image(img, 1423,1540, 2071,2280)
		txt=read_tesseract_image(img_crop)
		#print txt
		#print search_line("1/5", "2/5",1)
		find=False
		for str in search_list:
			if search_line(txt, str,1):
				find=True

		if find:
			print file_img


