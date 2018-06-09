#!/usr/bin/python3
# 2018.01.16 01:11:49 CST
# 2018.01.16 01:55:01 CST
import cv2
import numpy as np
import sys
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the image")
args = vars(ap.parse_args())

Image=cv2.imread(args["image"])
I=Image.copy()
img=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)

#Otsu Thresholding
img = cv2.GaussianBlur(img,(1,1),0)
ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(Image, contours, -1, (0,255,0), 3)

for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        if h>90:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(I, (x, y), (x + w, y + h), (255, 0, 255), 0)


cv2.imshow('edges',I)
cv2.waitKey(0)

"""
edges = cv2.Canny(gray,100,200,apertureSize = 3)
cv2.imshow('edges',edges)
cv2.waitKey(0)

minLineLength = 30
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('hough',img)
cv2.waitKey(0)

"""



## (1) read
"""
img = cv2.imread("bbb.jpg", 0)
img = cv2.GaussianBlur(img,(1,1),0)
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#img = cv2.drawContours(Image, contours, -1, (0,255,0), 3)

(im2, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
for contour in cnts[:2000]:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h/w
    area = cv2.contourArea(contour)
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2) 

cv2.imshow('aaa', thresh)
cv2.waitKey(0)
# You need to choose 4 or 8 for connectivity type
"""
#img2 = cv2.resize(img, (0,0), fx=2, fy=2)
#cv2.imshow('aaa', img2)
#cv2.waitKey(0)
#img = img.astype(np.uint8)
## (2) threshold

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""

th, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

image,contours, hierarchy = cv2.findContours(thresh, 1, 2)

for cnt in contours:
	x0,y0,w0,h0 = cv2.boundingRect(cnt)
	cv2.rectangle(thresh,(x0,y0),(x0+w0,y0+h0),(0,255,0),1)
#thresh = cv2.resize(thresh,(1000,50),interpolation = cv2.INTER_NEAREST)
cv2.imshow('aaa', thresh)
cv2.waitKey(0)
## (3) minAreaRect on the nozeros

pts = cv2.findNonZero(threshed)
ret = cv2.minAreaRect(pts)

print("aaaaaaaaaaaa", ret)

(cx,cy), (h, w), ang = ret
#print(w, h)
if w>h:
    w,h = h,w
    ang += 90

## (4) Find rotated matrix, do rotation
M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
rotated = cv2.warpAffine(threshed, M, (img2.shape[1], img2.shape[0]))

cv2.imshow('aaa', rotated)
cv2.waitKey(0)
## (5) find and draw the upper and lower boundary of each lines
hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

#print(hist)

th = 2
H,W = img2.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

#print(uppers)
#print(lowers)

rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

for y in lowers:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)

#rotated = rotated[0:200, 0:300]

cv2.imshow("result", rotated)
cv2.waitKey(0)
"""
