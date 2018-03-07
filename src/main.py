# for useability interface/ user input to quit display
import sys

import cv2

from VideoReader import VideoReader
from FindBall import BallFinder
from FindCourtCorners import CourtFinder


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
    done = False
    while(not(done)):
        ret, frame = vr.readFrame()
        ret2, frame2 = vr2.readFrame()
        if not(ret) or not(ret2):
            done = True
        else:
            frameDiff = bf.maskDiff(frame, frame2)
            findBall = bf.calcBallCenter(frameDiff)
            frame = bf.drawBallOnFrame(frame)
            if cf.found_corners:
                frame = cf.drawCornersOnFrame(frame)
            cv2.imshow('frame',cv2.resize(frame, (960, 540)))
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
