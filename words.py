from lines_splitter import *
from words_splitter import *
from char import *


def convert_image_to_words(path_to_file):
    lines_indexes = calculate_lines_indexes(path_to_file)
    words_list, words_number_in_line = split_lines_to_words(path_to_file, lines_indexes)

    return (words_list, words_number_in_line)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the image")
args = vars(ap.parse_args())
tab, new_lines = convert_image_to_words(args["image"])
text =""
line_index = 0
word_counter = 0
for word in tab:
	#cv2.imshow('img',word)
	string_word = predict(word)
	print (string_word)
	word_counter += 1
	if (new_lines[line_index] == word_counter):
		text += string_word + '\n'
		line_index += 1
		word_counter = 0
	else:
		text += string_word + " "
	#cv2.waitKey(1000)
print(text)
