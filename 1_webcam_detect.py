# !/usr/bin/env python

import cv2 as cv
import numpy
import scipy
import mtcnn

cam = cv.VideoCapture(1)

cv.namedWindow("test")
img_counter = 0

# create the detector, using default weights
detector = mtcnn.MTCNN()

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    faces = detector.detect_faces(frame)
    for face in faces:
#        print(face)
        x, y, width, height = face['box']
        color = (0,round(255*face['confidence'],0), 255)

        for key, coord in face['keypoints'].items():
            # create and draw dot
            dot = cv.circle(frame, coord, 2, color, 1)

        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels based on the confidence level
        cv.rectangle(frame, (x, y), (x2, y2), color, 1)

    cv.imshow("test", frame)

    #step out of the while loop before the video is done playing
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
#garbage collection
cam.release()
cv.destroyAllWindows()