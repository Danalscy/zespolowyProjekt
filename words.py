from lines_splitter import *
from words_splitter import *
from char import *


def convert_image_to_words(path_to_file):
    lines_indexes = calculate_lines_indexes(path_to_file)
    words_list = split_lines_to_words(path_to_file, lines_indexes)

    return words_list
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the image")
args = vars(ap.parse_args())
tab = convert_image_to_words(args["image"])
text =""
for word in tab:
	#cv2.imshow('img',word)
	string_word = predict(word)
	print (string_word)
	text += string_word + " "
	#cv2.waitKey(1000)
print(text)
