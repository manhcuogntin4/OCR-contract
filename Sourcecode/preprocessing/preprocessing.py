from rotation_normalization import *
from cropping import *
import shutil
import argparse
import os
import cv2
import re


if __name__ == '__main__':

    # Parse the input arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputDir', dest='dir_input', help='input directory with images to preprocess', required=True)
    parser.add_argument('--outputDir', dest='dir_output', help='output directory with images preprocessed', required=True)
    parser.add_argument('--errorDir', dest='dir_error', help='error directory with image preprocessed', required=True)
    options = parser.parse_args()

    # Initialize directories
    if not os.path.exists(options.dir_output):
        os.makedirs(options.dir_output)
    if not os.path.exists(options.dir_error):
        os.makedirs(options.dir_error)

    # Preprocessing
    for img_path in os.listdir(options.dir_input):
        try:
            full_img_path = os.path.abspath(os.path.join(options.dir_input, img_path))
            img_orig = cv2.imread(full_img_path)
            print full_img_path
            derotated_img, angle = rotation_normalization(img_orig)
            cropped_img = image_cropping(derotated_img)
            id_img = re.search('{[A-Z0-9-]*}', img_path).group(0)
            cv2.imwrite(os.path.join(options.dir_output, img_path + "_pp.png"), cropped_img)
        except Exception as e:
            print e.message
            shutil.move(full_img_path, options.dir_error)
            pass