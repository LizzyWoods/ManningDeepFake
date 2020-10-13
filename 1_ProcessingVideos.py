import os
import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot


basePath = "C:/Users/Pistidda.a/OneDrive - Hogeschool Leiden/data/DeepFake"
origFolder = "originalTIMIT"
fakeFolder = "deepfake TIMIT/higher_quality"
fakeFolder2 = "deepfake TIMIT/lower_quality"
dumpOutputFolder = "output/histogram"
videoFolder = vidFile = ""

#first I find the videoFolders:
for subDir in os.scandir(os.path.join(basePath,origFolder)):
    videoFolder = subDir.name
    print(videoFolder);

    # 1. Use Python Glob library to recursively iterate through both real and fake videos folders.
    aviFiles = glob.glob(os.path.join(basePath,origFolder,videoFolder,"*.avi"), recursive=True)

    # this prints out all *.avi files for that videoFolder
    # now we have to look up the corresponding fake video:
    for origVidFile in aviFiles:
        print(origVidFile)

        vidFile = os.path.splitext(os.path.basename(origVidFile))[0]

        #let's make a bold assumption there is only one matching DeepFake video.
        # in indexError => no DeepFake available, skip this origVidFile
        try:
            fakeVidFile = glob.glob(os.path.join(basePath, fakeFolder, videoFolder, str(vidFile) + "*.avi"))[0]
        except IndexError as error:
            print ("No DeepFake Video Available");
            continue

        print (fakeVidFile)

        outputFolder = os.path.join(basePath, dumpOutputFolder,videoFolder,vidFile)
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        # 2.a Loop through the frames of each video
        #   b. pick a video of choice
        origVid = cv.VideoCapture(origVidFile);
        fakeVid = cv.VideoCapture(fakeVidFile);
        if not (origVid.isOpened() & fakeVid.isOpened()):
            if (not origVid.isOpened()):
                print("Could not open video File: ",origVidFile)
            if (not fakeVid.isOpened()):
                print("Could not open video File: ",fakeVidFile)
            os._exit(0)

        cv.namedWindow("FrameCompare")
        frameNr = 0

        while (origVid.isOpened()):
            origRet, origFrame = origVid.read()
            fakeRet, fakeFrame = origVid.read()

            if (origRet & fakeRet):
                #show on the screen for visual feedback
                diff = cv.absdiff(origFrame, fakeFrame)
                merge = np.concatenate((origFrame, fakeFrame, diff), axis=1)
                cv.imshow("FrameCompare", merge)

                #convert to matlab colors for plotting all together
                origFrame = cv.cvtColor(origFrame, cv.COLOR_BGR2RGB)
                fakeFrame = cv.cvtColor(fakeFrame, cv.COLOR_BGR2RGB)
                diff = cv.cvtColor(diff, cv.COLOR_BGR2RGB)
        # 4.b. What does the difference look like?

        #write images and histogram to file:
                fig, plts = pyplot.subplots(2, 3)
                plts[0, 0].imshow(origFrame)
                plts[0, 1].imshow(fakeFrame)
                plts[0, 2].imshow(diff)

        # 4.a Compute histograms for original and deepfake and visualize them with matplotlib.
                plts[1, 0].plot(cv.calcHist([origFrame],[0],None,[256],[0,256]), color='red')
                plts[1, 0].plot(cv.calcHist([origFrame],[1],None,[256],[0,256]), color='green')
                plts[1, 0].plot(cv.calcHist([origFrame],[2],None,[256],[0,256]), color='blue')
                asp = np.diff(plts[1, 0].get_xlim())[0] / np.diff(plts[1, 0].get_ylim())[0]
                plts[1, 0].set_aspect(asp)

                plts[1, 1].plot(cv.calcHist([fakeFrame], [0], None, [256], [0, 256]), color='red')
                plts[1, 1].plot(cv.calcHist([fakeFrame], [1], None, [256], [0, 256]), color='green')
                plts[1, 1].plot(cv.calcHist([fakeFrame], [2], None, [256], [0, 256]), color='blue')
                asp = np.diff(plts[1, 1].get_xlim())[0] / np.diff(plts[1, 1].get_ylim())[0]
                plts[1, 1].set_aspect(asp)

                plts[1, 2].plot(cv.calcHist([diff], [0], None, [256], [0, 256]), color='red')
                plts[1, 2].plot(cv.calcHist([diff], [1], None, [256], [0, 256]), color='green')
                plts[1, 2].plot(cv.calcHist([diff], [2], None, [256], [0, 256]), color='blue')
                #asp = np.diff(plts[1, 2].get_xlim())[0] / np.diff(plts[1, 2].get_ylim())[0]
                #plts[1, 2].set_aspect(asp)
        # 3.a Save all frames as JPEG images for one original and the corresponding Deepfake video to the disk.
        #   b. pick a frame of choice

        # 5. Observe what makes video frames from Deepfake videos look different compared to originals.
                pyplot.savefig(os.path.join(outputFolder, "frame_" + str(frameNr) + ".png"), bbox_inches='tight')
                pyplot.close(fig)

                frameNr += 1

                #step out of the while loop before the video is done playing
                if cv.waitKey(25) & 0xFF == ord('q'):
                    fakeVid.release();
                    origVid.release();
                    cv.destroyAllWindows();
                    os._exit(0)
            else:
                break

        # When everything done, release the video capture object
        fakeVid.release();
        origVid.release();
        cv.destroyAllWindows();

        #end:while (origVid.isOpened()):
        #video is processed, next video

    #end: for origVidFile in aviFiles:
    #we finished processing all videos in the videoFolder

#end for subDir in os.scandir(os.path.join(basePath,origFolder)):
#done with the program


