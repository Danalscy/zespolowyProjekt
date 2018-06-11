import numpy as np
from char import *
from sklearn.externals import joblib
def test(hog):
    print('Testowanie modelu ... ')
    clf = joblib.load('alphabet.pkl')
    root = "errorData"
    data = []
    labels_accuracy = [0] * 63
    count_images = [0] * 63
    paths = []
    
    for path, subdirs, files in os.walk(root):
        for name in files:
                pathToImg = os.path.join(path, name)
                dir_int = (os.path.split(os.path.dirname(pathToImg))[1])
                dir_int = int(dir_int.replace('Sample',''))
                print (pathToImg + " "),
                #print  (dir_int)

                ch = getValueOf(dir_int)
                img = cv2.imread(pathToImg,0)
                img = processImg(img)
                descriptor = getFeatures(hog,img)
                descriptor = np.transpose(descriptor)
                predicted = clf.predict(descriptor)
                character = getValueOf(predicted)
                if (character == ch):
                	labels_accuracy[dir_int] += 1
                count_images[dir_int] += 1
    print (labels_accuracy)
    print (count_images)
    for i in range(1,len(labels_accuracy)):
    	char = getValueOf(i)
    	labels_accuracy[i] = labels_accuracy[i]/count_images[i]
    	print (char + ": " + str(labels_accuracy[i]))
    	#print (str(labels_accuracy[i]))

hog = getHog()
test(hog)
