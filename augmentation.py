from audioop import reverse
import cv2
from cv2 import floodFill
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


img_RGB = cv2.imread("C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/candle/test/good/0038.JPG",cv2.IMREAD_COLOR)
img = cv2.imread("C:/Users/MVCLAB/Desktop/DataSet/VisA_20220922/candle/test/good/0038.JPG",0)
img_RGB = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA)
cv2.imshow('gray',img)
cv2.waitKey(0) 
print(type(img))

ret,thresh1 = cv2.threshold(img,70,225,cv2.THRESH_BINARY_INV) #50

cv2.imshow('test',thresh1)
cv2.waitKey(0) 

kernel = np.ones((3,3), np.uint8) 
dilation_img = cv2.dilate(thresh1, kernel, iterations = 1)

cv2.imshow('close',dilation_img)
cv2.waitKey(0) 

gray_lap = cv2.Laplacian(dilation_img, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(gray_lap) # 轉回uint8

cv2.imshow('test',dst)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(dst, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

mx = (0,0,0,0)      # biggest bounding box so far
mx_area = 0
for cont in contours:
    x,y,w,h = cv2.boundingRect(cont)
    area = w*h
    if area > mx_area:
        mx = x,y,w,h
        mx_area = area
x,y,w,h = mx



# cnt =contours[1]
# bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
# for bbox in bounding_boxes:
#   [x,y,w,h] = bbox
#   # cv2.rectangle(img_RGB, (x, y), (x+w, y+h), (0, 0, 225), 0)
#   cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 225), 0)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 225), 0)

cv2.imshow('ROI',img)
cv2.waitKey(0)
   
# cropped = img_RGB[y+1:y+h,x+1:x+w]
cropped = img_RGB[y+1:y+h,x+1:x+w]

   
cv2.imshow('cropped', cropped)
key = cv2.waitKey(0)
# 按空白鍵
if key == 32:   # ASCII Code
  cv2.destroyAllWindows()
# 按's'存圖
elif key == ord('s'):
  cv2.imwrite('cropped_image.jpg', cropped)
  cv2.destroyAllWindows()


