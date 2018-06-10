from lines_splitter import *
from words_splitter import *

def convert_image_to_words(path_to_file):
    lines_indexes = calculate_lines_indexes(path_to_file)
    words_list = split_lines_to_words(path_to_file, lines_indexes)

    return words_list

