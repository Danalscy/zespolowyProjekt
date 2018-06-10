import cv2
import numpy as np

def count_min_line_height(uppers, lowers):
    sum = 0
    for i in range(len(uppers)):
        sum += lowers[i] - uppers[i]
    return sum / len(uppers) / 2


def filter_lines_for_height(uppers, lowers):
    min_line_height = count_min_line_height(uppers, lowers)
    zipped_lists = list(zip(uppers[:], lowers[:]))
    for i, (u, l) in enumerate(zipped_lists):
        if ((l - u) < min_line_height):
            uppers.remove(u)
            lowers.remove(zipped_lists[i-1][1])
    return (uppers, lowers)

def filter_incomplete_lines(uppers, lowers):
    if(len(uppers) > 1 and len(lowers) > 1):
        if(lowers[0] < uppers[0]):
            lowers.pop(0)
        if(lowers[-1] < uppers[-1]):
            uppers.pop(-1)
    return (uppers, lowers)

def prepare_lines_indexes(uppers, lowers):
    complete_uppers, complete_lowers = filter_incomplete_lines(uppers, lowers)
    final_uppers, final_lowers = filter_lines_for_height(complete_uppers, complete_lowers)
    return (final_uppers, final_lowers)


def calculate_lines_indexes(path_to_file):
    img2 = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
    th, threshed = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)

    (cx,cy), (w, h), ang = ret
    if (ang < -45):
        ang += 90

    M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
    rotated = cv2.warpAffine(threshed, M, (img2.shape[1], img2.shape[0]))

    hist = cv2.reduce(rotated,1, cv2.REDUCE_AVG).reshape(-1)

    th = 1
    H,W = img2.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

    uppers, lowers = prepare_lines_indexes(uppers, lowers)

    return list(zip(uppers,lowers))

