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

    filename = '../UntrackedFiles/angle3_5.mp4'
    vr = VideoReader(filename)
    # find court corners:
    cf = CourtFinder()
    numFrames = vr.getNumFrames()
    vr.setNextFrame(int(numFrames/2))
    ret, frame = vr.readFrame()
    cf.FindCourtCorners(frame, 0)

    # reset frame index to beginning
    vr.setNextFrame(0)
    # having 2 video readers is much much faster than resetting next frame id
    vr2 = VideoReader(filename)
    bf = BallFinder()

    numFrames = int(vr.getNumFrames())
    vr2.setNextFrame(5) # for background subtraction, 5 frames ahead-ish

    # initialize
    numVRs = 5;
    vrs = [VideoReader(filename) for i in range(0,numVRs)];
    for i in range(0,numVRs):
        vrs[i].setNextFrame(i);


    done = False
    while(not(done)):
        rets = [];
        frames = [];
        for i in range(0, numVRs):
            ret, frame = vrs[i].readFrame();
            if not(ret):
                done = True
            print 'FRAME SHAPE', frame.shape
            frames.append(frame);
            rets.append(frame);

        avgFrame = calcAvgFrame(frames);


        # trying some stuff to make ball finding more robust
        # absolute frame difference:
        frameDiff = frames[0].astype(np.int16) - avgFrame.astype(np.int16)
        frameDiff = abs(frameDiff)
        frameDiff = frameDiff.astype(np.uint8)
        # threshold difference
        frameDiff = cv2.cvtColor(frameDiff, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(frameDiff, 50, 255)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
        mask_filt = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask_filt = cv2.morphologyEx(mask_filt, cv2.MORPH_DILATE, se)
        ___notsure, contours, hier = cv2.findContours(mask_filt,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        mask_ecc = np.zeros(mask_filt.shape, dtype="uint8");
        for c in contours:
            cArea = cv2.contourArea(c)
            if len(c) < 5: # too small to measure eccenetricity
                cv2.drawContours(mask_ecc,[c],0,255,-1)
                continue;
            c_ellipse = cv2.fitEllipse(c)
            rect = c_ellipse[1];
            ecc = np.max([rect[0]/(rect[1]+0.001), rect[1]/(rect[0]+0.001)]);
            if ecc < 2 and cArea < 150:
                cv2.drawContours(mask_ecc,[c],0,255,-1)

        colorFiltMask = bf.hsvFilt(frames[0], False);
        mask_filt = colorFiltMask & mask_filt;

        # connectedComps = cv2.connectedComponentsWithStats(mask, 8)
        # stats = connectedComps[2]
        # print 'stats ', stats
        frameFilt = frames[0] & np.dstack((mask,mask,mask))
        cv2.imshow('frameDiff',cv2.resize(mask_filt, (960, 540)))
        #cv2.imshow('frameDiff', frameDiff)
        cv2.waitKey(1)



        #print "press q enter to quit "
        #done = False
        #while(not(done)):
        #    c = sys.stdin.read(1)
        #    if c == 'q':
        #       done = True
        #return

        # frameDiff = bf.maskDiff(frame, frame2, False)
        # findBall = bf.calcBallCenter(frameDiff)
        # frame = bf.drawBallOnFrame(frame)
        # if cf.found_corners:
        #     frame = cf.drawCornersOnFrame(frame)
        # cv2.imshow('frame',cv2.resize(frameDiff, (960, 540)))
        # cv2.waitKey(1)

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
