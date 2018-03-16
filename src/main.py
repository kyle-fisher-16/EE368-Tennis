# for useability interface/ user input to quit display
import sys

import cv2
import numpy as np

from VideoReader import VideoReader
from FindBall import BallFinder
from FindCourtCorners import CourtFinder


def calcAvgFrame(frames):
    numFrames = len(frames);
    sumFrame = np.zeros(frames[0].shape, dtype="uint32");
    print 'SUM FRAME SHAPE:', sumFrame.shape
    for i in range(0, numFrames):
        sumFrame += frames[i];
    sumFrame /= numFrames;
    return sumFrame.astype(np.uint8);

def main():

    filename = '../UntrackedFiles/stereoClip11_Kyle.mov'
    vr = VideoReader(filename)
    # find court corners:
    cf = CourtFinder()
    numFrames = int(vr.getNumFrames())
    vr.setNextFrame(int(numFrames/2))
    ret, frame = vr.readFrame()
    cf.FindCourtCorners(frame, 0)

    # reset frame index to beginning
    vr.setNextFrame(0)
    bf = BallFinder()

    # second video reader for faster processing
    vr2 = VideoReader(filename)
    numFrameForward = 5
    vr2.setNextFrame(numFrameForward)

    # # calculate average frame for background
    # numFrameForAve = 10;
    # ret, sumFrame = vr.readFrame()
    # sumFrame = sumFrame.astype(np.int16)
    # for i in range(1,numFrameForAve):
    #     ret, frame = vr.readFrame()
    #     sumFrame += frame.astype(np.int16)
    # avgFrame = sumFrame/numFrameForAve
    # avgFrame = avgFrame.astype(np.uint8)
    #
    # # reset video reader index to beginning again
    # vr.setNextFrame(0)

    done = False
    while(not(done)):
        ret, frame = vr.readFrame()
        ret2, frame2, = vr2.readFrame()
        if not(ret) or not(ret2):
            done = True
        else:
            frameDiff1 = bf.hsvDiff(frame, frame2)
            frameDiff2 = bf.rgbDiff(frame, frame2)
            frameCornerMask = bf.GetCornernessMask(frame, frame2)
            mask = bf.aveMask(frameDiff1, frameDiff2, frameCornerMask)
            cv2.imshow('frame',cv2.resize(frameDiff2, (960, 540)))
            cv2.waitKey(1)

    # cv2.imwrite('../UntrackedFiles/frame_diff.jpg', frame_diff)
    # # quit sequence:
    # print "press q enter to quit "
    # done = False
    # while(not(done)):
    #     c = sys.stdin.read(1)
    #     if c == 'q':
    #        done = True

    vr.close()

if __name__ == '__main__':
    main()
