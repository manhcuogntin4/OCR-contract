import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('CA3.pdf-4.png',0)

template = cv2.imread('template.png',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    match=img_rgb[pt[1]:pt[1]+h, pt[0]:pt[0] + w]
    cv2.imwrite('match.png',match) 
    #print pt[1], pt[1]+h, pt[0], pt[0] + w 
    print pt[1], pt[0]
cv2.imwrite('res.png',img_rgb)
cv2.imshow('res', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()