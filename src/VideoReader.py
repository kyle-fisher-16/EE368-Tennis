# class for reading video mp4 files
# hopefully then we won't have to write a million of the same functions

# imports for reading videos:
import numpy as np
import cv2 as cv

# note to self: opencv reads images in and stores as BGR, not RGB!!!

class VideoReader(object):
    # opens video files and fills in useful parameters
    def __init__(self, filename = 'video.mp4'):
        self.fn = filename
        # self.fn = '../UntrackedFiles/clip25.mp4'
        self.vid = cv.VideoCapture(self.fn)
        # hopefully useful variables?
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.framerate = self.vid.get(cv.CAP_PROP_FPS)
        # print self.framerate
        self.numFrames = self.vid.get(cv.CAP_PROP_FRAME_COUNT)

    # make sure to close video capture when done!
    def close(self):
        self.vid.release()
        cv.destroyAllWindows()

    def readFrame(self):
        if (self.vid.isOpened()):
            # ret is a boolean for if there are still frames left
            # ret = false if there are no frames left or disconnected/no frame to return
            # frame is None if at end or if error/no frames returned
            ret, frame = self.vid.read()
            return ret, frame
        else:
            print 'video file not opened'
            return False, None

    def playVideo(self):
        done = False
        while(not(done)):
                ret, frame = self.readFrame()
                done = False
                c = cv.waitKey(1) & 0xFF
                if  c == ord('q') or not(ret):
                    done = True
                    return
                    # return done
                cv.imshow('frame', frame)
                # return done



    def getNumFrames(self):
        return self.numFrames

    def getNextFrameIdx(self):
        # 0-indexed index of next frame to be captured
        return self.vid.get(cv.CAP_PROP_POS_FRAMES)

    def setNextFrame(self, idx):
        # make sure idx is in bounds:
        if idx >= 0 and idx <= self.numFrames:
            self.vid.set(cv.CAP_PROP_POS_FRAMES, idx)
        else:
            print 'error: index out of frame range'
