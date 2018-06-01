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
reload(sys)
sys.setdefaultencoding('utf-8')
this_dir = os.path.dirname(__file__)

def create_folder(filename):
	
	path=os.path.basename(filename)
	base_name=os.path.splitext(path)[0]
	if not os.path.exists(base_name):
		os.makedirs(base_name)
	return base_name


def move_file_folder(filename, folder):
	filename_new=os.path.join(folder, filename)
	shutil.move(filename, filename_new)


filename="CA3.pdf"
folder_path=creat_folder(filename)
move_file_folder(filename,folder_path)




