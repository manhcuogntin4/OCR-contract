import pyslibtesseract
import cv2
tesseract_config = pyslibtesseract.TesseractConfig(psm=pyslibtesseract.PageSegMode.PSM_AUTO, hocr=True)
f=open("ocr.html","w")
ocr=pyslibtesseract.LibTesseract.simple_read(tesseract_config, 'CA5.pdf-1.png')
f.write(ocr)
f.close()