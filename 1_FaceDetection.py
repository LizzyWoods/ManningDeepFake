import os
import glob
import cv2 as cv
import numpy as np
import mtcnn

basePath = "C:/Users/Pistidda.a/OneDrive - Hogeschool Leiden/data/DeepFake"
origFolder = "originalTIMIT"
fakeFolder = "deepfake TIMIT/higher_quality"
fakeFolder2 = "deepfake TIMIT/lower_quality"
dumpOutputFolder = "output/faceDetection"
videoFolder = "faks0"
vidFile = "si1573"
classifierFile = "haarcascade_frontalface_default.xml"

fakeVidFile = fakeVidFile = None
#get the 2 corresponding files
try:
    origVidFile = glob.glob(os.path.join(basePath, origFolder, videoFolder, str(vidFile) + "*.avi"))[0]
    fakeVidFile = glob.glob(os.path.join(basePath, fakeFolder, videoFolder, str(vidFile) + "*.avi"))[0]
except IndexError as error:
    print("No Video Available");
    os._exit(0)

outputFolder = os.path.join(basePath, dumpOutputFolder, videoFolder, vidFile)
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

print(origVidFile);
print(fakeVidFile);

#load the classifier: MTCNN classifier
classifier = cv.CascadeClassifier(classifierFile)

# 2.a Load a video and loop through the frames
origVid = cv.VideoCapture(origVidFile);
fakeVid = cv.VideoCapture(fakeVidFile);
if not (origVid.isOpened() & fakeVid.isOpened()):
    if (not origVid.isOpened()):
        print("Could not open video File: ",origVidFile)
    if (not fakeVid.isOpened()):
        print("Could not open video File: ",fakeVidFile)
    os._exit(0)

cv.namedWindow("CroppedFaceOriginal")
cv.namedWindow("CroppedFaceDeepfake")

frameNr = 0
originalROI = deepfakeROI = None
while (origVid.isOpened()):
    origRet, origFrame = origVid.read()
    fakeRet, fakeFrame = origVid.read()

    if (origRet & fakeRet):
        merge = np.concatenate((origFrame, fakeFrame), axis=1)
# 2.    Use Viola-Jones method provided in OpenCV to detect faces for each video frame of a given video.
#       You can also use more advanced tools, for instance,
#       the pre-trained MTCNN model is popular and has a higher accuracy of face detection.

#       detect all faces in the picture:
        bboxes = classifier.detectMultiScale(merge)

        boxNr = 0
        for box in bboxes:
            # draw a rectangle over the pixels
            x, y, width, height = box
            x2, y2 = x + width, y + height

            if x < width:
                #original face
                originalROI = merge[y:y2,x:x2]
                cv.imshow("CroppedFaceOriginal",originalROI)
                cv.rectangle(merge, (x, y), (x2, y2), (0,0,255), 1)
            else:
                #deepfake face
                deepfakeROI = merge[y:y2, x:x2 ]
                cv.imshow("CroppedFaceDeepfake", deepfakeROI)
                cv.rectangle(merge, (x, y), (x2, y2), (0,255,255), 1)

            boxNr += 1
        cv.imshow("FrameCompare", merge)

        frameNr += 1

    #step out of the while loop before the video is done playing
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
fakeVid.release();
origVid.release();
cv.destroyAllWindows();

#end:while (origVid.isOpened()):
#1 video is processed, next video


