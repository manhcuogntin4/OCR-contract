import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("CA3.pdf-1_1.png",0)
ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
idx=imgBinary<1
cv2.imwrite("binary.png",imgBinary)
count= np.sum(idx)
print count