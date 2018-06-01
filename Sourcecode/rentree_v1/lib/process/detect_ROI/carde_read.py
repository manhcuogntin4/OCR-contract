# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
from fuzzyset import FuzzySet
import difflib

import re
import unicodedata
 


def read_tesseract_image(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    text=strip_accents(text)
    corpus = [line.strip() for line in text.split("\n")]
    return corpus

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

def search_line(corpus):
	check=False
	for line in corpus:
		words=line.split()
		print words
		for word in words:
			d=levenshteinDistance("Adhesion", word)
			if d<=2:
				return line
def check_image(corpus, str_search, threshold=5):
    for line in corpus:
        #print line
        d=levenshteinDistance(line, str_search)
        #print d
        if d<=threshold:
            return True
    return False

def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in xrange(ord(c1), ord(c2)+1):
        yield chr(c)
def getID(line):
	ID=""
	for char in line:
		if char in char_range('0','9'):
			ID+=char
	return ID
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image to be OCR'd")
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
        help="type of preprocessing to be done")
    args = vars(ap.parse_args())

    # load the example image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     
    # check to see if we should apply thresholding to preprocess the
    # image
    if args["preprocess"] == "thresh":
        gray = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
     
    # make a check to see if median blurring should be done to remove
    # noise
    elif args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)
     
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)

    #print(text)
    text=strip_accents(text)
    #print text

    corpus = [line.strip() for line in text.split("\n")]
    #print corpus
    line=search_line(corpus)
    print line
    print getID(line)


