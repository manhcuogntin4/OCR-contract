import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *

img_list=read_file_folder("./", "*.png")
template = cv2.imread('template.png',0)

for file_img in img_list:
	img_rgb=cv2.imread(file_img,0)
	#template = cv2.imread('template.png',0)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where( res >= threshold)
	#print res
	if len(loc) >0:
		if len(loc[1])>0:
			print loc[1]
			print file_img