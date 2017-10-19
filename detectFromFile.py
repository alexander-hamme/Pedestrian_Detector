
"""
Created on Sat Mar 15 11:29:51 2017

@author: Alexander Hamme
With inspiration from Adrian Rosebrock's PyImageSearch blog
"""

# USAGE: python detect.py --image image

from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import cv2
import time

# construct the argument parse and parse the arguments

DEFAULT_MIN_IMAGE_WIDTH = 400
DEFAULT_IMAGE_PADDING = (16, 16)
SHOW_ORIGINAL_IMG = False

CALIBRATION_MODE_1 = (400, (3, 3), (16, 16), 1.01, 0.999)  # People very small size and close together
CALIBRATION_MODE_2 = (400, (3, 3), (32, 32), 1.01, 0.8)  # People very small size
CALIBRATION_MODE_3 = (400, (4, 4), (32, 32), 1.01, 0.999)  # People small size and close together
CALIBRATION_MODE_4 = (400, (4, 4), (32, 32), 1.01, 0.8)  # People small size
CALIBRATION_MODE_5 = (400, (4, 4), (32, 32), 1.02, 0.999)  # People medium size and close together
CALIBRATION_MODE_6 = (400, (4, 4), (32, 32), 1.02, 0.8)  # People medium size
CALIBRATION_MODE_7 = (400, (6, 6), (32, 32), 1.035, 0.999)  # People large size and close together
CALIBRATION_MODE_8 = (400, (6, 6), (32, 32), 1.035, 0.8)  # People large size

CALIBRATION_MODES = (CALIBRATION_MODE_1, CALIBRATION_MODE_2, CALIBRATION_MODE_3, CALIBRATION_MODE_4, CALIBRATION_MODE_5,
                     CALIBRATION_MODE_6, CALIBRATION_MODE_7, CALIBRATION_MODE_8)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")

ap.add_argument("-v", "--detection_values", required=True, nargs=3, help=(
                                                                    "'-v winstride scale overlap'\n"
                                                                    "|Winstride is a number around 4 +/- 3)"
                                                                    "|Scale is a number between 1.01 and 1.5"
                                                                    "|Overlap (Threshold) is a number between 0.6 and 1")
                )

ap.add_argument("-m", "--calibration_mode", type=int, required=False, help=(
                                                                    "Number between 1 and 8 inclusive|\n"
                                                                    "Mode 1: People very small and close together|\n"
                                                                    "Mode 2: People very small|\n"
                                                                    "Mode 3: People small and close together|\n"
                                                                    "Mode 4: People small|\n"
                                                                    "Mode 5: People medium size and close together|\n"
                                                                    "Mode 6: People medium size|\n"
                                                                    "Mode 7: People large size and close together|\n"
                                                                    "Mode 8: People large size|\n")
                )

args = vars(ap.parse_args())

# HOG descriptor/person detector
t = time.time()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

image = cv2.imread(args["image"])
values = args["detection_values"]
mode = args["calibration_mode"]

if mode:
    MIN_IMAGE_WIDTH, WIN_STRIDE, PADDING, SCALE, OVERLAP_THRESHOLD = CALIBRATION_MODES[int(mode) - 1]
else:
    values = ((int(values[0]), int(values[0])), float(values[1]), float(values[2]))
    WIN_STRIDE, SCALE, OVERLAP_THRESHOLD = values
    MIN_IMAGE_WIDTH, PADDING = DEFAULT_MIN_IMAGE_WIDTH, DEFAULT_IMAGE_PADDING

image = imutils.resize(image, width=min(MIN_IMAGE_WIDTH, image.shape[1]))
orig = image.copy()

# detect people in the image
(rects, wghts) = hog.detectMultiScale(image, winStride=WIN_STRIDE,
                                      padding=PADDING, scale=SCALE)

# apply non-maxima suppression to the bounding boxes, but use a fairly large overlap threshold, 
# to try to maintain overlapping boxes that are separate people

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=OVERLAP_THRESHOLD)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    # tighten the boxes by a small margin
    shrinkW, shrinkH = int(0.15 * xB), int(0.05*yB)
    cv2.rectangle(image, (xA, yA), (xB-shrinkW, yB-shrinkH), (0, 255, 0), 2)

print("{} people detected in image.\nElapsed time: {} seconds".format(len(pick), int((time.time() - t) * 100) / 100.0))

if SHOW_ORIGINAL_IMG:
    cv2.imshow("Original Image", orig)
cv2.imshow("After Detection", image)
cv2.waitKey(0)
