# import the necessary packages
import numpy as np
import argparse
import glob
import cv2
 
def auto_canny(image, sigma=0.1):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to input dataset of images")
ap.add_argument("-o", "--output", required=True,
    help="path to output dataset of images")
args = vars(ap.parse_args())
 
# loop over the images
for i, imagePath in enumerate(glob.glob(args["images"] + "/*.jpg")):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    # wide = cv2.Canny(blurred, 10, 200)
    # tight = cv2.Canny(blurred, 225, 250)
    # auto = auto_canny(blurred, sigma=125.)
    auto = cv2.Canny(blurred, 50, 75)
    # Invert colors
    auto = (auto - 255)*255

    savepath = args['output'] + '/' + imagePath.split('/')[-1]

    cv2.imwrite(savepath, auto)