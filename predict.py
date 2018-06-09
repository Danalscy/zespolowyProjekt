from char import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to the image")
args = vars(ap.parse_args())
predict(args["image"])

