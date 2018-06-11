import cv2
import numpy as np

def split_lines_to_words(path_to_file, lines_indexes):
    list_of_words = []
    words_number_in_line = []

    img = cv2.imread(path_to_file)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((100,15), np.uint8)

    for upp, down in lines_indexes:
        words_in_line = split_line_to_words(img[upp:down,:], kernel)
        words_number_in_line.append(len(words_in_line))
        for word in words_in_line:
       	    list_of_words.append(word)

    return (list_of_words, words_number_in_line)

def split_line_to_words(line, kernel):
    list_of_words = []
    #cv2.imshow('line',line)
    #cv2.waitKey(0)
    ret, thresh = cv2.threshold(line,127,255,cv2.THRESH_BINARY_INV)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    im2, ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = line[y:y+h, x:x+w]
        list_of_words.append(roi)
        
        #cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)

    return list_of_words

