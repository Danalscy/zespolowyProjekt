import numpy as np
from char import *
from sklearn.externals import joblib
def train(hog):
    print('Ładowanie danych ... ')
    root = "trainingData"
    data = []
    labels = []
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
                #cv2.imshow('img',img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                descriptor = getFeatures(hog,img)
                data.append(descriptor)
                labels.append(dir_int)

    x,y = len(data),len(data[0])
    labels = np.array(labels)
    data = np.squeeze(data)
   
    clf = svm.SVC()
    print('Uczenie modelu ... ')
    clf.fit(data, labels)
    joblib.dump(clf, 'alphabet.pkl')

hog = getHog()
train(hog)
