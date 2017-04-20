from __future__ import print_function
from imutils.object_detection import non_max_suppression
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import argparse
import imutils
import cv2
import time


class LineDetector:
    CAMERA_RESOLUTION = (1024, 768)
    BOX_COLOR = (0, 255, 0)
    CALIBRATION_MODE_1 = (400, (3, 3), (32, 32), 1.01, 0.999)   # People very small size and close together
    CALIBRATION_MODE_2 = (400, (3, 3), (32, 32), 1.01, 0.8)     # People very small size
    CALIBRATION_MODE_3 = (400, (4, 4), (32, 32), 1.015, 0.999)   # People small size and close together
    CALIBRATION_MODE_4 = (400, (4, 4), (32, 32), 1.015, 0.8)     # People small size
    CALIBRATION_MODE_5 = (400, (4, 4), (32, 32), 1.02, 0.999)   # People medium size and close together
    CALIBRATION_MODE_6 = (400, (4, 4), (32, 32), 1.02, 0.8)     # People medium size
    CALIBRATION_MODE_7 = (400, (8, 8), (32, 32), 1.03, 0.999)  # People large size and close together
    CALIBRATION_MODE_8 = (400, (8, 8), (32, 32), 1.03, 0.8)    # People large size
    CALIBRATION_MODES = (CALIBRATION_MODE_1, CALIBRATION_MODE_2, CALIBRATION_MODE_3, CALIBRATION_MODE_4,
                         CALIBRATION_MODE_5, CALIBRATION_MODE_6, CALIBRATION_MODE_7, CALIBRATION_MODE_8)

    def __init__(self):
        # default mode
        self.MIN_IMAGE_WIDTH, self.WIN_STRIDE, self.PADDING, self.SCALE, self.OVERLAP_THRESHOLD = self.CALIBRATION_MODE_5
        self.SHOW_IMAGES = True
        self.IMAGE_WAIT_TIME = 0  # Wait indefinitely until button pressed
        self.GRAY_SCALE = False

    def get_picture(self):
        '''
        Take a single picture from default video stream
        :return: numpy.ndarray
        '''

        ''' On a non Raspberry Pi computer, use the following code instead:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        '''

        # Use With statements to close and flush PiCamera output stream
        with PiCamera() as camera:
            with PiRGBArray(camera) as output:
                camera.resolution = self.CAMERA_RESOLUTION
                camera.capture(output, format="bgr")
                image = output.array
                output.truncate(0)  # Flushes and empties camera between captures
        
        if self.SHOW_IMAGES:
            cv2.imshow('Original Image', image)
            cv2.waitKey(self.IMAGE_WAIT_TIME)
            cv2.destroyAllWindows()

        if self.GRAY_SCALE:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def set_calibration(self, idx=None, tup=None):
        '''
        Set people detection calibration with EITHER an index of a preset calibration mode, or manually set all values
        with a tuple. If passed both an index and a tuple, only the index will be considered.
        :param idx: index of a calibration mode in self.CALIBRATION_MODES
        :param tup: alternative to idx, tuple with 5 values
        :return: None
        '''
        if idx and idx < len(self.CALIBRATION_MODES):
            self.MIN_IMAGE_WIDTH, self.WIN_STRIDE, self.PADDING, self.SCALE, self.OVERLAP_THRESHOLD = self.CALIBRATION_MODES[idx]
        elif tup:
            assert len(tup) == 5
            self.MIN_IMAGE_WIDTH, self.WIN_STRIDE, self.PADDING, self.SCALE, self.OVERLAP_THRESHOLD = tup

    def find_people(self, img):
        '''
        Detect people in image
        :param img: numpy.ndarray
        :return: count of rectangles after non-maxima suppression, corresponding to number of people detected in picture
        '''
        t = time.time()
        # HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Chooses whichever size is less
        image = imutils.resize(img, width=min(self.MIN_IMAGE_WIDTH, img.shape[1]))
        # detect people in the image
        (rects, wghts) = hog.detectMultiScale(image, winStride=self.WIN_STRIDE,
                                              padding=self.PADDING, scale=self.SCALE)
        # apply non-maxima suppression to the bounding boxes but use a fairly large overlap threshold,
        # to try to maintain overlapping boxes that are separate people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=self.OVERLAP_THRESHOLD)

        print("Elapsed time: {} seconds".format(int((time.time() - t) * 100) / 100.0))

        if self.SHOW_IMAGES:
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                # Tighten the rectangle around each person by a small margin
                shrinkW, shrinkH = int(0.05 * xB), int(0.15*yB)
                cv2.rectangle(image, (xA+shrinkW, yA+shrinkH), (xB-shrinkW, yB-shrinkH), self.BOX_COLOR, 2)

            cv2.imshow("People detection", image)
            cv2.waitKey(self.IMAGE_WAIT_TIME)
            cv2.destroyAllWindows()
            
        return len(pick)

def test():
    detector = LineDetector()
    for i in range(3):
        print (["Three", "Two", "One"][i])
        time.sleep(1)
    detector.find_people(detector.get_picture())

