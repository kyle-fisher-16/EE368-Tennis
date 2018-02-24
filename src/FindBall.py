# script to take a video file and recognize balls
# for now only ball candidates, but utlimately only moving balls

# import video reader class. File in same folder (src)
from VideoReader import VideoReader
# still need opencv for other image processing stuff
import cv2 as cv

# for useability interface/ user input to quit display
import sys

def main():
    vr = VideoReader('../UntrackedFiles/clip25.mp4')
    ret, frame = vr.readFrame()
    cv.imshow('frame',frame)
    c = cv.waitKey(1) & 0xFF    # necessary to get image to display

    # now to do the image processing... to be later made into nice class and functions

    # quit sequence:
    print "press q enter to quit "
    done = False
    while(not(done)):
        c = sys.stdin.read(1)
        if c == 'q':
            done = True

    #vr.playVideo()
    vr.close

if __name__ == '__main__':
    main()
