#OpenCV python HOG feature based people detector applied on video
#Find the peopledetect.py on opencv-master/samples/python on your OpenCV installation

import numpy as np
from imutils.object_detection import non_max_suppression
import cv2
passTotal = 0

def draw_detections(img, rects, thickness = 1):
    global passTotal
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for x, y, w, h in pick:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        passagem = len(pick)
        passTotal += passagem
        #passagem = str(len(pick))
        print("Passagem Total: " + str(passTotal))

if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap=cv2.VideoCapture('video2.mp4')
    while True:
        _,frame=cap.read()
        frame2 = frame[100:150, 0:850]
        found,w=hog.detectMultiScale(frame2, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame2,found)
        cv2.imshow('feed',frame2)
        ch = 0xFF & cv2.waitKey(30)
        if ch == 27:
           break
    cv2.destroyAllWindows()
