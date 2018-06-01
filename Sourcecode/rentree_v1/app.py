# -*- coding: utf-8 -*-
import os
import cv2
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
from tools.axademo import detect_cni
from tools.rentre_signee import detect_rentree
import os
import caffe
import glob
import copy
#Multiprocess
from multiprocessing import Pool   
import multiprocessing.pool
import multiprocessing 
import subprocess
from multiprocessing import Manager
from functools import partial


#Mulitprocess with child process
from multiprocessing.pool import Pool as PoolParent
from multiprocessing import Process
import time
#Process Unicode here
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

#Process image with textcleaner
import subprocess
# Process similar
from difflib import SequenceMatcher
import _init_paths
from detect_ROI.utils import *
from detect_ROI.cosine_similar import find_lines_similar
from detect_ROI.carde_read import check_image, read_tesseract_image

CAFFE_ROOT ='/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
MODELE_DIR = os.path.join(REPO_DIR, 'models/googlenet')
DATA_DIR = os.path.join(REPO_DIR, 'data/googlenet') 
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
# To do
def convert_contract_to_folder(filename):
	print "Convert contract to folder"
	file_list=convert_file_folder(filename)
	return file_list
#filename="test/test_no/n2.pdf"
#filename="test/re3.pdf"
#filename="test/re2.pdf"
filename="test/re4.tif"
filename=os.path.join(REPO_DIR, filename)
file_list=convert_file_folder(filename)
#file_path="/tmp/caffe_demos_uploads/re1/re1.tif-1.png"
# if check_image(file_path, "BULLETIN D'ADHESION/ CERTIFICAT D'ADHESION"):
# 	print "found"
paragraph=[]
page1=""
count=0
page1_path=""
score=0
page_signature_path=""
score_signature=0
for file_path in file_list:
	corpus=read_tesseract_image(file_path)
	# if check_image(corpus, "a parapher",3):
	# 	count+=1
	# 	print "page1 a 6", file_path
	# 	paragraph.append(file_path)

	if check_image(corpus, "BULLETIN D'ADHESION/ CERTIFICAT D'ADHESION"):
		print "page1", file_path
        line,sc=find_lines_similar(corpus, "BULLETIN D'ADHESION/ CERTIFICAT D'ADHESION")
        if sc>score:
            page1_path=file_path
            score=sc

	if check_image(corpus, "Modèle de lettre de renonciation", 8):
		print "page_signature", file_path
		line,sc=find_lines_similar(corpus, "Modèle de lettre de renonciation")
		if sc>score_signature:
			page_signature_path=file_path
			score_signature=sc



        #break
#for file_path in file_list:
	# if check_image(corpus, "a parapher",3):
	# 	count+=1
	# 	print "page1 a 6", file_path
	# 	paragraph.append(file_path)
	# if check_image(corpus, "Modele de lettre de renonciation", 5):
	# 	print "page signature", file_path

print page1_path, page_signature_path
numero_adhesion, page_counter_clstm, rature= detect_rentree(page1_path,1)
numero_page1=get_numero_filename(get_base_name(page1_path))

print page_signature_path
C41_remplir,C42_check,C43_check= detect_rentree(page_signature_path,2)
numero_page_signature=get_numero_filename(get_base_name(page_signature_path))

if C41_remplir:
	str_remplir="C41 est rempli"
else:
	str_remplir="C41 n'est pas rempli"

if rature:
	str_rasture="Rature"
else:
	str_rasture="Non rature"

if C42_check:
	str_C42signature="C42 signé"
else:
	str_C42signature="C42 non signé"
if C43_check:
	str_C43signature="C43 signé"
else:
	str_C43signature="C43 non signé"

print numero_page1, numero_adhesion, rature
print numero_page_signature, C41_remplir,C42_check,C43_check
if page_counter_clstm!="":
	number_page=int(page_counter_clstm[-1])
	n=numero_page_signature-numero_page1+1
	if n>number_page:
		n=n/2
	print number_page, n

#print paragraph

print "Result", str_rasture,str_remplir, str_C42signature, str_C43signature




# for file_path in file_list:
# 	detect_rentree(file_path)
#print file_list






