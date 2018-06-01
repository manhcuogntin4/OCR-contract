from PIL import Image
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def display_image(image, gray=True):
    plt.figure(figsize=(12, 6))
    if gray == True:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()


def custom_barplot(height, title):
    x = np.arange(len(height))
    plt.bar(x, height=height)
    plt.title(title)
    plt.grid(True)
    plt.show()


def get_sum(img):
    x_sum = np.sum(img, axis=0) / img.shape[1]
    y_sum = np.sum(img, axis=1) / img.shape[0]
    return x_sum, y_sum

def distrib(img):
    x_sum, y_sum = get_sum(img)
    custom_barplot(y_sum, "Distrib pixel axe Y")
    #custom_barplot(x_sum, "Dis

def find_y_index(sum_histo, bound):
	for index,s in enumerate(sum_histo):
		if index>bound[0][0] and index<200 and np.sum(sum_histo[index-10:index+10])<50:
			return index


def is_close_gap(tuple1, tuple2, treshold=100):
    # Check if gap between two spikes is small
    if tuple2[0] - tuple1[1] < treshold:
        return True
    else:
        return False

def get_boundaries(tuple_array):
    # Remove small gap between spikes
    tmp = tuple_array[0]
    bound = []
    for i in range(len(tuple_array)):
        if i+1 == len(tuple_array):
            break
        if is_close_gap(tuple_array[i], tuple_array[i+1]):
            tmp = (tmp[0], tuple_array[i+1][1])
        else:
            bound.append(tmp)
            tmp = tuple_array[i+1]
    bound.append(tmp)
    return bound

def get_biggest_boundary(tuple_array):
    # Keep only the largest boundary
    diff = tuple_array[0][1] - tuple_array[0][0]
    tmp = tuple_array[0]
    for item in tuple_array:
        if diff < item[1] - item[0]:
            tmp = item
            diff = item[1] - item[0]
    return tmp
def get_variation(array):
    # get spikes variations
    bounds = []
    b1 = 0
    b2 = 0
    flag = True
    for i, pixel in enumerate(array):
        if pixel > 4:
            if flag:
                b1 = i
                flag = False
        if pixel < 5:
            if not flag:
                b2 = i-1
                bounds.append((b1, b2))
                flag = True
    if b2 == 0 or b1 != bounds[-1][0]:
        bounds.append((b1, len(array)-1))
    return bounds

img = cv2.imread('test2.png',0)

img_binary=cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
distrib(cv2.adaptiveThreshold(img_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2))

x_sum,y_sum=get_sum(img_binary)
y_var = get_variation(y_sum)
y_bound=get_boundaries(y_var)
print y_bound[0][0]
print img.shape[0]
index=find_y_index(y_sum,y_bound)
print index
mask=np.full(img.shape,255)
h,w=img.shape
img_signature=img[:index,:]
mask[0:index,0:w]=img_signature
cv2.imwrite("signature.png",img_signature)
cv2.imwrite("mask.png",mask)