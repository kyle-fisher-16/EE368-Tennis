# script to take a video file and recognize balls
# for now only ball candidates, but utlimately only moving balls

# import video reader class. File in same folder (src)
from VideoReader import VideoReader
# still need opencv for other image processing stuff
import cv2

# need numpy
import numpy as np

# for useability interface/ user input to quit display
import sys

class BallFinder(object):
    def __init__(self):
        # hsv in opencv: h [0,179], s [0,255], v [0,255]
        self.ballColHsv = np.array([int(44), int(115), int(250)])
        # far away balls have very small saturation levels
        satThresh = 75;
        valThresh = 5;
        hueThresh = 20;
        self.lowerCol = self.ballColHsv - [hueThresh, satThresh, valThresh]
        self.upperCol = self.ballColHsv + [hueThresh, satThresh, valThresh]
        # centroid of found ball in motion for tracking
        self.ballPixelLoc = [0,0]
        # maybe we should try like kalman filtering this or something?

    # returns the binary frame difference between frames 1 and 2
    # use to locate the ball initially where frame2 for background subtraction
    # set withFilt to False if don't want to perform opening to get rid of noise
        # if ball is far away it is very small and
        # may be smaller than noise in foreground
    def maskDiff(self, frame1, frame2, withFilt=True):
        hsvFrame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsvFrame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsvFrame1, self.lowerCol, self.upperCol)
        mask2 = cv2.inRange(hsvFrame2, self.lowerCol, self.upperCol)
        # remember that these are uints, so no negative values
        # xor to get 2 distinct balls, then and with original to only get first ball
        frameDiff = mask1 ^ mask2
        frameBall1 = frameDiff & mask1
        if withFilt:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            frameBall1Filt = cv2.morphologyEx(frameBall1, cv2.MORPH_OPEN, se)
            return frameBall1Filt
        else:
            return frameBall1

    # calculate the centroid of a ball where frame_mask is a binary frame
    # sets self.ballPixelLoc and returns if ball was found or not
    def calcBallCenter(self, frameMask):
        # find connected components to extract the location of the ball(s)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 8
        # Perform the operation
        connectedComps = cv2.connectedComponentsWithStats(frameMask, connectivity)
        # Get the results
        # The first cell is the number of labels
        numLabels = connectedComps[0]
        # The second cell is the label matrix
        labels = connectedComps[1]
        # The third cell is the stat matrix
        stats = connectedComps[2]
        # The fourth cell is the centroid matrix
        centroids = connectedComps[3]

        #largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # background is largest component with label 0
        # ball should be next largest component with label 1
        foundBall = False
        print numLabels
        if numLabels > 1:
            self.ballPixelLoc = np.around(centroids[1]).astype(int)
            foundBall = True
        else:
            print 'No ball found in frame'

        return foundBall

    # puts a circle of ball size at the last known ball location
    # returns the edited frame
    def drawBallOnFrame(self, frame):
        # put circle on top of ball in original frame
        # make the circle pink
        #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
        ballFrame = cv2.circle(frame, tuple(self.ballPixelLoc), 10, (180,105,255), -1)
        return ballFrame
