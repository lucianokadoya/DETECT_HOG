#OpenCV python HOG feature based people detector applied on video
#Find the peopledetect.py on opencv-master/samples/python on your OpenCV installation

import numpy as np
from imutils.object_detection import non_max_suppression
import cv2

passTotal = 0
file = raw_input("Entre com o arquivo: ")
reg1 = input("Entre com a altura inicial de deteccao: ")
reg2 = input("Entre com a altura final de deteccao: ")
reg3 = input("Entre com a largura inicial de deteccao: ")
reg4 = input("Entre com a largura final de deteccao: ")

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
    cap=cv2.VideoCapture(str(file))
    while True:
        _,frame=cap.read()
        frame2 = frame[reg1:reg2, reg3:reg4]
        found,w=hog.detectMultiScale(frame2, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame2,found)
        cv2.imshow('feed',frame2)
        ch = 0xFF & cv2.waitKey(30)
        if ch == 27:
           break
    cv2.destroyAllWindows()
