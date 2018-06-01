import numpy as np
import scipy
from scipy.stats import norm
import pandas as pd
import sys,os
import time
import cPickle
import datetime
import math

import caffe
CAFFE_ROOT = '/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
#os.chdir(caffe_root)
MODELE_DIR=os.path.abspath(os.path.dirname(__file__))
class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/deploy.prototxt'.format(MODELE_DIR)),
        'pretrained_model_file': (
            '{}/rentree_signee_check_signature.caffemodel'.format(MODELE_DIR)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(CAFFE_ROOT)),
        'class_labels_file': (
            '{}/synset_words.txt'.format(MODELE_DIR)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(CAFFE_ROOT)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 256
    default_args['raw_scale'] = 255.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)
        self.net = net
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        # transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
        self.transformer = transformer
        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort('synset_id')['name'].values

    def classify_image(self, image):
        try:
            net = self.net
            net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

            starttime = time.time()
            # scores = self.net.predict([image], oversample=True).flatten()
            out = net.forward()
            proba = out['prob'][0]
            scores = net.blobs['fc8'].data[0]
            print proba, scores
            endtime = time.time()

            indices = (-proba).argsort()[:3]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta_proba = [
                (p, '%.5f' % proba[i])
                for i, p in zip(indices, predictions)
            ]

            score = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            print score, meta_proba
            return proba, predictions
            #return (True, score, meta_proba, '%.3f' % (endtime - starttime))

        except Exception as err:
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    #print ss
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    print ss
    pvar = 1.0*ss/n # the population variance
    print n, pvar
    return pvar**0.5

def classify_signature(filename, gpu=0):
    img=caffe.io.load_image(filename)
    ImagenetClassifier.default_args.update({'gpu_mode': gpu})
    classifieur=ImagenetClassifier(**ImagenetClassifier.default_args)
    return classifieur.classify_image(img)

# filename='/home/cuong-nguyen/2017/Workspace/Fevrier/CodeSource/AnnotationTool/AnnotationTool/python/document_category_googlenet/test/re5.png'
# result= classify_signature(filename)
# proba_list = result[2]
# print proba_list
#result= classify_signature(filename)
# if __name__ == '__main__':
#     caffe_root = '/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
#     REPO_DIR = os.path.abspath(os.path.dirname(__file__))
#     sys.path.insert(0, caffe_root + 'python')



     

#     net_file=os.path.join(REPO_DIR,'deploy.prototxt')
#     print net_file

#     caffe_model=os.path.join(REPO_DIR,'rentree_signee_check_signature.caffemodel')
#     #caffe_model='/home/cuong-nguyen/2017/Workspace/Fevrier/CodeSource/AnnotationTool/AnnotationTool/python/document_category_googlenet/train_val_nouveau_permis_03_09.caffemodel'

#     mean_file='/home/cuong-nguyen/2016/Workspace/brexia/Decembre/CodeSource/codeHaoming/web-demo-version4.0/caffe-fast-rcnn/python/caffe/imagenet'

     

#     net = caffe.Net(net_file,caffe_model,caffe.TEST)

#     transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#     transformer.set_transpose('data', (2,0,1))

#     #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))

#     transformer.set_raw_scale('data', 255)

#     transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead if BGR

     

#     #im=caffe.io.load_image(caffe_root+'examples/images/cat.jpg')

#     img=caffe.io.load_image('/home/cuong-nguyen/2017/Workspace/Fevrier/CodeSource/AnnotationTool/AnnotationTool/python/document_category_googlenet/test/re5.png')

#     #img=caffe.io.load_image('/home/cuong-nguyen/2017/Workspace/Fevrier/Documents/prefix-1.png')

#     net.blobs['data'].data[...] = transformer.preprocess('data',img)

#     out = net.forward()
#     print out 

#     proba = out['prob'][0]
#     scores = net.blobs['fc8'].data[0]
#     print proba, scores
#     #top_k= out['fc8'][0]

#     imagenet_labels_filename =os.path.join(REPO_DIR,'synset_words.txt')

#     labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

#     print labels 

