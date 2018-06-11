from sklearn.externals import joblib
from PIL import Image
from numpy import*
import numpy as np
import cv2 
import sys
import argparse
from common import clock, mosaic

from skimage import transform as tf
import glob
import os
import os.path
import cv2

from sklearn import svm
SZ = 32

def bbox2(matrix):
    rows = np.any(matrix, axis=1)
    cols = np.any(matrix, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    new_matrix=matrix[rmin:rmax,cmin:cmax]  
    resized=tf.resize(new_matrix,(50,35),order = 0)
    return resized
def normalize(vector):
    mini = np.amin(vector)
    maxi = np.amax(vector)
    for i in range(len(vector)):
    	vector[i]=np.rint((vector[i]-mini)/(maxi-mini) * 10000)
    return vector
"""
def printMatrix(matrix):
    for i in range(0,len(matrix)):
    	for j in range(0,len(matrix[0])):
    		print int(matrix[i][j]),
    	print
"""
def imgToBinMatrix(img): 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (thresh, threshold) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('threshold',threshold)
    A = array(threshold)             
    matrix=empty((A.shape[0],A.shape[1]),None)    

    for i in range(0,len(threshold)):
    	for j in range(0,len(threshold[0])):
    		if threshold[i][j]==255:
    			matrix[i][j]=0
    		else:
    			matrix[i][j]=1
    return matrix
def apply_pca(train_images,components):
    pca = PCA(n_components=components)
    train_images = pca.fit_transform(train_images)
    return train_images

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img.copy()
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
def getHog():
    winSize = (32,32)
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    signedGradients = True
    #im = deskew(img)
    #hog = cv2.HOGDescriptor()
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    return hog

def getFeatures(hog,img):
    digits_deskewed = map(deskew, img)
    descriptor = hog.compute(img)
    return descriptor

def processImg(img):    
    img = cv2.GaussianBlur(img,(3,3),0)
    #img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    #ret,thresh = cv2.threshold(img,127,255,0)
    (ret, thresh) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image,contours, hierarchy = cv2.findContours(thresh, 1, 2)
    
    maxSize = 0 
    x=y=w=h=0
    """ 
    for cnt in contours:
    	#M = cv2.moments(cnt)
    	x0,y0,w0,h0 = cv2.boundingRect(cnt)
    	if ((w0+h0) > maxSize) and ((w0+h0) < img.shape[0] +img.shape[1]):
    		maxSize = w0 + h0
    		x,y,w,h = x0,y0,w0,h0
    	#cv2.rectangle(img,(x0,y0),(x0+w0,y0+h0),(0,255,0),2)
    	#if not((w == 0) or (h==0)): 
    	
    img = img[y-3:y+h+3,x-3:x+w+3]    
    """
    #print (pathToImg) 
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    
    img = cv2.resize(img,(SZ,SZ),interpolation = cv2.INTER_NEAREST)
    #cv2.imshow('edges',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img

def predict(img):
        
    clf = joblib.load('alphabet.pkl') 
    #img = cv2.imread(pathToImg,0)
    #I = cv2.imread(pathToImg)
    hog = getHog()    

    #cv2.imshow('word',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    img = cv2.GaussianBlur(img,(1,1),0)
    ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    image, contours, hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours,bounding = sort_contours(contours)
    img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    ascii_word = "" 
    for contour in contours:
    	[x, y, w, h] = cv2.boundingRect(contour)
    	#print (x,y,w,h)	
    	if (w*h>200): 
    		crop_img = img[y:y+h,x:x+w]
    		#cv2.imshow('Letter',crop_img)
    		crop_img = img[y-3:y+h+3,x-2:x+w+2]
    		crop_img = processImg(crop_img)
    		#cv2.imshow('Letter',crop_img)
    		descriptor = getFeatures(hog,crop_img);
    		descriptor = np.transpose(descriptor)
    		predicted = clf.predict(descriptor)
    		#print (getValueOf(predicted))
    		ascii_word += getValueOf(predicted)

    		#cv2.waitKey(0)
    		#cv2.destroyAllWindows()
        	# draw rectangle around contour on original image
    		#cv2.rectangle(img, (x, y), (x + w , y + h ), (0, 255, 0), 1)
    		#cv2.putText(img,getValueOf(predicted),(int((x+x+w)/2),y+h+15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),3)
    
    return ascii_word

def sort_contours(cnts):
    reverse = False
    i = 0

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b:b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)
    """
    cv2.imshow('edges',I)
    cv2.waitKey(0)
   
    img = processImg(img)
    #imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    cv2.imshow('Letter',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

   

    descriptor = getFeatures(hog,img);
    descriptor = np.transpose(descriptor)
    print (descriptor.shape)
    print('Predykcja znaku ... ')
    predicted = clf.predict(descriptor)
    print (predicted, getValueOf(predicted))
    
    
    model = cv2.ml.SVM_create()
    model.setGamma(0.50625)
    model.setC(12.5)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    #samples = np.array(descriptor)
    #labels = np.array(1)
    #data = data.reshape(-1,)
    descriptor = np.matrix(descriptor)
    descriptor = np.reshape(descriptor,-1)
    #for i in descriptor:
    #	print (i)
    #print (labels)
    #print (descriptor.shape)
    print('Uczenie modelu ... ')
    model.train(data, cv2.ml.ROW_SAMPLE, labels)
    
    print('Predykcja znaku ... ')
    predicted = model.predict(descriptor)[1].ravel()
    character = getValueOf(predicted)
    print (predicted, character)
    #print('Evaluating model ... ')
    #vis = svmEvaluate(model, digits_test, hog_descriptors_test, labels_test)
    
    #mbin = imgToBinMatrix(img)
    #vector = getAttributes(matrix)
    #letters = pd.read_csv('data.txt')
    #printMatrix(matrix)
    #print (vector)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #(thresh, threshold) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    hog_descriptors = []
    for img in digits_deskewed:
        hog_descriptors.append(hog.compute(img))
    hog_descriptors = np.squeeze(hog_descriptors)
    

    
    pca_images = apply_pca(matrix,0.7)
    print pca_images
    print pca_images.shape
    data =[]
    labels=[]
    data.append(pca_images)
    print data
    print data.shape
    labels.append(11)
    clf = svm.svc(kernel='rbf',c=10,gamma=0.1)
    clf.fit(data, labels)
    
    #training_points = np.array(letters.drop(['letter'], 1))
    #training_labels = np.array(letters['letter'])
    
    training_points = np.float32(training_points)
    training_labels = np.intc(training_labels)
    print (training_points.shape)
    print (training_labels.shape)
    #training_points =  training_points.astype(float)
    kNearest = cv2.ml.KNearest_create()
    kNearest.train(training_points,training_labels)
    retval, npaResults, neigh_resp, dists = kNearest.findNearest(vector,k=1)    
    
    
    clf = SVC()
    clf.fit(training_points, training_labels)
    #expected = test_labels
    predicted = clf.predict(vector)
    predicted_letter = getValueOf(predicted) 
    print predicted, predicted_letter
    """
    
def getValueOf(f):
		if f == 1:
			ch = '0'
		elif f == 2:
			ch = '1'
		elif f == 3:
			ch = '2'
		elif f == 4:
			ch = '3'
		elif f == 5:
			ch = '4'
		elif f == 6:
			ch = '5'
		elif f == 7:
			ch = '6'
		elif f == 8:
			ch = '7'
		elif f == 9:
			ch = '8'
		elif f == 10:
			ch = '9'
		elif f == 11:
			ch = 'A'
		elif f == 12:
			ch = 'B'
		elif f == 13:
			ch = 'C'
		elif f == 14:
			ch = 'D'
		elif f == 15:
			ch = 'E'
		elif f == 16:
			ch = 'F'
		elif f == 17:
			ch = 'G'
		elif f == 18:
			ch = 'H'
		elif f == 19:
			ch = 'I'
		elif f == 20:
			ch = 'J'
		elif f == 21:
			ch = 'K'
		elif f == 22:
			ch = 'L'
		elif f == 23:
			ch = 'M'
		elif f == 24:
			ch = 'N'
		elif f == 25:
			ch = 'O'
		elif f == 26:
			ch = 'P'
		elif f == 27:
			ch = 'Q'
		elif f == 28:
			ch = 'R'
		elif f == 29:
			ch = 'S'
		elif f == 30:
			ch = 'T'
		elif f == 31:
			ch = 'U'
		elif f == 32:
			ch = 'V'
		elif f == 33:
			ch = 'W'
		elif f == 34:
			ch = 'X'
		elif f == 35:
			ch = 'Y'
		elif f == 36:
			ch = 'Z'
		elif f == 37:
			ch = 'a'
		elif f == 38:
			ch = 'b'
		elif f == 39:
			ch = 'c'
		elif f == 40:
			ch = 'd'
		elif f == 41:
			ch = 'e'
		elif f == 42:
			ch = 'f'
		elif f == 43:
			ch = 'g'
		elif f == 44:
			ch = 'h'
		elif f == 45:
			ch = 'i'
		elif f == 46:
			ch = 'j'
		elif f == 47:
			ch = 'k'
		elif f == 48:
			ch = 'l'
		elif f == 49:
			ch = 'm'
		elif f == 50:
			ch = 'n'
		elif f == 51:
			ch = 'o'
		elif f == 52:
			ch = 'p'
		elif f == 53:
			ch = 'q'
		elif f == 54:
			ch = 'r'
		elif f == 55:
			ch = 's'
		elif f == 56:
			ch = 't'
		elif f == 57:
			ch = 'u'
		elif f == 58:
			ch = 'v'
		elif f == 59:
			ch = 'w'
		elif f == 60:
			ch = 'x'
		elif f == 61:
			ch = 'y'
		elif f == 62:
			ch = 'z'	
		elif f == 63:
			ch = ','
		elif f == 64:
			ch = ';'
		elif f == 65:
			ch = ':'
		elif f == 66:
			ch = '?'
		elif f == 67:
			ch = '!'
		elif f == 68:
			ch = '.'
		elif f == 69:
			ch = '@'
		elif f == 70:
			ch = '#'
		elif f == 71:
			ch = '$'
		elif f == 72:
			ch = '%'
		elif f == 73:
			ch = '&'
		elif f == 74:
			ch = '('
		elif f == 75:
			ch = ')'
		elif f == 76:
			ch = '{'
		elif f == 77:
			ch = '}'
		elif f == 78:
			ch = '['
		elif f == 79:
			ch = ']'
		else: 
			ch = ''

		return ch
def getAttributes(oldMatrix):
    #matrix = bbox2(oldMatrix)
    #matrix=tf.resize(oldmatrix,(20,20),order = 0)
    #x_box = cmin
    #y_box = matrix.shape[0]-rmax
    #width = cmax - cmin
    #high = rmax - rmin
    matrix = oldMatrix
    width = matrix.shape[0]
    high = matrix.shape[1]
    onpix = np.sum(matrix)
    midx,midy = ((width)/2 , (high)/2)
#   print rmin,rmax,cmin,cmax
    x_bar = x2bar = y_bar = y2bar = xybar = x2ybr = xy2br = x_ege = xegvy = y_ege = yegvx = 0.0
    #print matrix
    #print midx, midy
    #columnSums = np.sum(matrix, axis=0)
    #print columnSums
    for i in range(0,len(matrix)):
    	for j in range(0,len(matrix[0])):
    		if matrix[i][j]==1:
    			x_bar = x_bar + (j - midx)/width
    			y_bar = y_bar + (midy - i)/high
    			x2bar = x2bar + ((j - midx) * (j - midx))/width
    			y2bar = y2bar + ((midy - i) * (midy - i))/high

    			x_part0 = (j - midx)/width
    			y_part0 = (midy - i)/high
    			xybar = xybar + (x_part0 * y_part0)
			
    			x_part1 = (j - midx) * (j - midx)/width
    			y_part1 = (midy - i)/high
    			x2ybr = x2ybr + (x_part1 * y_part1)

    			x_part2 = (j - midx)/width
    			y_part2 = (midy - i) * (midy -i)/high
    			xy2br = xy2br + (x_part2 * y_part2)
			
    			if(j-1<0):	
    				x_ege = x_ege + 1
    				xegvy = xegvy + ((high - i)  - midy)
    			elif (matrix[i][j-1] == 0):
    				x_ege = x_ege + 1
    				xegvy = xegvy + ((high - i)  - midy)
    			if(i + 1 == len(matrix)):		
    				y_ege = y_ege + 1
    				yegvx = yegvx + (j - midx)
    			elif (matrix[i+1][j] == 0):
    				y_ege = y_ege + 1
    				yegvx = yegvx + (j - midx)

    
    x_bar = (x_bar)/onpix
    y_bar = (y_bar)/onpix
    x2bar = x2bar/onpix
    y2bar = y2bar/onpix
    xybar = xybar/onpix
    x2ybr = x2ybr/onpix
    xy2br = xy2br/onpix
    x_ege = x_ege/width
    y_ege = y_ege/high

    np.set_printoptions(suppress = True)

    vector = np.array([onpix,x_bar,y_bar,x2bar,y2bar,xybar,x2ybr,xy2br,x_ege,xegvy,y_ege,yegvx])
    #print x_box,y_box,width,high,onpix,x_bar,y_bar,x2bar,y2bar,xybar,x2ybr,xy2br,x_ege,xegvy,y_ege,yegvx
    #print (vector)
    #vector = normalize(vector)
    vector =  vector.reshape(1,-1)
    return vector


