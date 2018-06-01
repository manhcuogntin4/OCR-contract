# -*- coding: utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Copyright (c) 2016 Haoming Wang
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#from ocr.clstm import clstm_ocr
#from ocr.clstm import clstm_ocr_calib
from ocr.clstm_rentree import clstm_ocr_rentree
from detect_ROI.carde_read import *
from detect_line.check_rasture import *
from detect_ROI.cosine_similar import check_filled_box
from classify.rentree_signee_check_signature import classify_signature
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import werkzeug
import datetime
import math
import pytesseract
from PIL import Image
from correct_skew import find_angle_skew
from PIL import Image as imp
from scipy.ndimage import interpolation as inter
import imutils

CLASSES = ('__background__', # always index 0
                         'C11','C12','C41','C42','C43','C1','C2','C3')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'inria': ('INRIA_Person',
                  'INRIA_Person_faster_rcnn_final.caffemodel'),
        'axa': ('axa_poc', #'cni_5k_faster_rcnn.caffemodel')}

                 'rentree_signee_15_05.caffemodel')}
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'

def vis_detections1(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def vis_detections(im, rois, thresh=0.5):
    """Draw detected bounding boxes."""

    #im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    # class_name, dets,
    for cls_ind, dets in enumerate(rois):
        class_name = CLASSES[cls_ind + 1] # avoid class 'background'
        if(cls_ind==5):# Prenom
            inds = np.where(dets[:, -1] >= 0.01)[0] 
        else:
            inds = np.where(dets[:, -1] >= thresh)[0]

        print inds
        if len(inds) == 0:
            continue
        maxp, ind_maxp = 0, 0

        for i in inds:
            score = dets[i, -1]
            if score > maxp:
                maxp, ind_maxp = score, i

        score = dets[ind_maxp, -1]
        #if(cls_ind==5):
        print score, cls_ind
        bbox = dets[ind_maxp, :4]
        bbox = map(int, bbox)
        cv2.imwrite(class_name + '.png', im[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        if class_name=="mrz":
            angle=find_angle_skew("mrz.png")
            if angle!=0:
                cni=cv2.imread("cni.png")
                rotated = imutils.rotate_bound(cni, -angle)
                cv2.imwrite('skew_corrected.png', rotated)
                #data = inter.rotate(im, angle, reshape=False, order=0)
                #img_corrected = imp.fromarray((255 * data).astype("uint8")).convert("RGB")
                #img_corrected.save('skew_corrected.png')
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def extract_roi(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    regions = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return regions

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # a small regulation of detected zone, comment me if the lastest result is good enough
        hight = bbox[3] - bbox[1]
        if class_name == 'C11':
            # bbox[0] += 1.5 * hight
            # bbox[1] -= 0.25 * hight
            # bbox[2] += 0.2 * (bbox[2] - bbox[0])
            # bbox[3] += 0.15 * hight
            print 'C11'
        elif class_name == 'C12':
            bbox[2] -= 0.2 * (bbox[2] - bbox[0])
            print 'C12'
        elif class_name == 'C41':
            print 'C41'
        elif class_name == 'C42':
            print 'C42'
        elif class_name == 'C43':
            print 'C43'
        elif class_name == 'C1':
            print 'C1'
        elif class_name == 'C2':
            print 'C2'
        elif class_name == 'C3':
            print 'C3'
        pts = [int(bx) for bx in bbox]
        regions.append(pts)
    return regions


def ocr_queue(im, bbx, cls, q):
	q.put(calib_roi(im,bbx,cls))



def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(UPLOAD_FOLDER, 'demo', image_name)
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    print im_file
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    rois = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        rois.append(dets)
    vis_detections(im, rois, thresh=CONF_THRESH)


def check_type_1(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(UPLOAD_FOLDER, 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    res = {}
    roi_file_name=[]
    rature=False
    numero_adhesion=""
    page_counter=""
    page_counter_clstm=0
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background, 'carte'
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        tmp = extract_roi(cls, dets, thresh=CONF_THRESH)
        if len(tmp) > 0:
            #bbx = tmp[0]  # TODO: Find the zone with greatest probability
            #txt, prob = clstm_ocr(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls=='lieu')
            #res[cls] = (bbx, txt, prob)
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
            bbx = tmp[0]  # TODO: Find the zone with greatest probability
                # txt, prob = clstm_ocr(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls=='lieu')
                # if(prob<0.95):
                #     txt1,prob1=clstm_ocr(im[bbx[1]-3:bbx[3]+3, bbx[0]:bbx[2]], cls=='lieu')
                #     if(prob<prob1):
                #         txt=txt1
                #         prob=prob1
          
            pts_msz = [int(bx) for bx in bbx]
            filename_ = werkzeug.secure_filename('output' + cls + image_name + '.png')
            filename = os.path.join(UPLOAD_FOLDER, filename_)
            cls_im=im[pts_msz[1]:pts_msz[3], pts_msz[0]:pts_msz[2]]
            cv2.imwrite(filename, cls_im)
            if cls=='C11':
                text = pytesseract.image_to_string(Image.open(filename))
                os.remove(filename)

                #print(text)
                text=strip_accents(text)
                #print text

                corpus = [line.strip() for line in text.split("\n")]
                #print corpus
                line=search_line(corpus)
                numero_adhesion= getID(line)
            if cls=='C12':
                print 'C12'
                rature=is_rasture(cls_im,1000)
            if cls=='C3':
                print 'C3'
                page_counter = pytesseract.image_to_string(Image.open(filename))
                print page_counter
                img = cv2.imread(filename,1)
                page_counter_clstm,prob=clstm_ocr_rentree(img)
                print page_counter_clstm, prob
    return numero_adhesion, page_counter_clstm, rature

def check_type_2(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(UPLOAD_FOLDER, 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    res = {}
    roi_file_name=[]
    C41_remplir=False
    C42_signature=False
    C43_signature=False
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background, 'carte'
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        tmp = extract_roi(cls, dets, thresh=CONF_THRESH)
        if len(tmp) > 0:
            #bbx = tmp[0]  # TODO: Find the zone with greatest probability
            #txt, prob = clstm_ocr(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls=='lieu')
            #res[cls] = (bbx, txt, prob)
        # vis_detections(im, cls, dets, thresh=CONF_THRESH)
            bbx = tmp[0]  # TODO: Find the zone with greatest probability
                # txt, prob = clstm_ocr(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls=='lieu')
                # if(prob<0.95):
                #     txt1,prob1=clstm_ocr(im[bbx[1]-3:bbx[3]+3, bbx[0]:bbx[2]], cls=='lieu')
                #     if(prob<prob1):
                #         txt=txt1
                #         prob=prob1
          
            pts_msz = [int(bx) for bx in bbx]
            filename_ = werkzeug.secure_filename('output' + cls + image_name + '.png')
            filename = os.path.join(UPLOAD_FOLDER, filename_)
            cls_im=im[pts_msz[1]:pts_msz[3], pts_msz[0]:pts_msz[2]]
            cv2.imwrite(filename, cls_im)
            if cls=='C41':
                print "C41 check remplir"
                check_str="Fait a , le Signature"
                check_str_2="Fait a , le"
                C41_remplir=check_filled_box(filename,check_str,0.6) and  check_filled_box(filename,check_str_2,0.6)
            if cls=='C42':
                print 'C42 check signature'
                C42_signature= classify_signature(filename)
            if cls=='C43':
                print 'C43 check signature'
                C43_signature= classify_signature(filename)
    return C41_remplir,C42_signature,C43_signature
    
    

def check(boxes, scores, thresh=0.8, nms_thresh=0.3):
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        if cls_ind == 5:  #we skip 'nom epouse' check
            continue
        if len(inds) == 0:
            return False
    return False



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        default=True,
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='axa')

    args = parser.parse_args()

    return args


def calib_roi(im,bbx,cls):
    txt, prob = clstm_ocr_parallel(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls=='lieu')
    cv2.setNumThreads(0)
    if(prob<0.95):
        for i in range(0,2):
            for j in range(0,2):
                txt_temp,prob_temp=clstm_ocr_calib(im[bbx[1]-5*i*math.pow( -1, j):bbx[3]+5*i*math.pow( -1, j), bbx[0]-3*i*math.pow( -1, j):bbx[2]+3*i*math.pow( -1, j)], cls=='lieu')
                if(prob<prob_temp):
                    txt=txt_temp
                    prob=prob_temp
    return txt, prob

def detect_rentree(filename, type=1):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.MODELS_DIR = '/home/cuong-nguyen/2017/Workspace/Fevrier/CodeSource/FasterRCNN/py-faster-rcnn/models'
    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')

    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for CA...'
    #return demo(net, filename)
    if type==1:
        return check_type_1(net, filename)
    if type==2:
        return check_type_2(net, filename)

def main():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.MODELS_DIR = '/home/cuong-nguyen/2017/Workspace/Fevrier/CodeSource/FasterRCNN/py-faster-rcnn/models'
    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    #im_name = 'ID_FRA.jpg'
    im_name = 'test_re1.png'     
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for data/demo/{}'.format(im_name)
    demo(net, im_name)

    plt.show()




if __name__ == '__main__':
    main()
