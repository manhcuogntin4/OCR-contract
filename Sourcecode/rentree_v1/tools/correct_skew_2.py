import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

input_file = sys.argv[1]




def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    print "data", data
    hist = np.sum(data, axis=1)
    print "hist 1", hist[1:]
    print "hist -1", hist[:-1]
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score





# correct skew
data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
img.save('skew_corrected.png')

def find_angle_skew(input_file):
	img = im.open(input_file)
	# convert to binary
	wd, ht = img.size
	pix = np.array(img.convert('1').getdata(), np.uint8)
	bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
	plt.savefig('binary.png')
	delta = 1
	limit = 30
	angles = np.arange(-limit, limit+delta, delta)
	scores = []
	for angle in angles:
	    hist, score = find_score(bin_img, angle)
	    scores.append(score)

	best_score = max(scores)
	best_angle = angles[scores.index(best_score)]
	print('Best angle: {}'.format(best_angle))
	return best_angle