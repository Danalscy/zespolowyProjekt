import glob
import os
import os.path
import cv2
import numpy as np
from char import *
from sklearn import svm
root = "trainingData"
data =[]
labels=[]
for path, subdirs, files in os.walk(root):
    for name in files:
        pathToImg = os.path.join(path, name)
	dir_int = (os.path.split(os.path.dirname(pathToImg))[1]) 
	dir_int = int(dir_int.replace('Sample',''))
	print pathToImg + " ",
        print  dir_int
	
	ch = getValueOf(dir_int)
	#print pathToImg + " in folder " + ch
	
        img = cv2.imread(pathToImg)
	hog = getHog()
	descriptor = getFeatures(hog,img)
	descriptor = np.insert(descriptor,0,dir_int)
	#descriptor = descriptor.reshape(-1,1)
	print (descriptor)
	print (descriptor.shape)
        data.append (descriptor)

#data = np.squeeze(data)
#print data
#print data.shape
np.savetxt('data.txt',data,delimiter =',',fmt = '%s')

