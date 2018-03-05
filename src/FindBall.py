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
        # means that only ball in first frame will appear
        # if want balls in both frames, use bitwise xor
        frameDiff = mask1 - mask2
        if withFilt:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            frameDiffFilt = cv2.morphologyEx(frameDiff, cv2.MORPH_OPEN, se)
            return frameDiffFilt
        else:
            return frameDiff

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
        print numLabels, centroids

        # draw on mask...
        centroidFrame = np.array(frameMask)
        for i in range(0,numLabels):
            center = np.around(centroids[i]).astype(int)
            centroidFrame = cv2.circle(centroidFrame, tuple(center), 10, (180,105,255), -1)
        cv2.imshow('frame',cv2.resize(frameMask, (960, 540)))
        cv2.waitKey(1)

        #largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # background is largest component with label 0
        # ball should be next largest component with label 1
        cnts = cv2.findContours(frameMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        foundBall = False
        if len(cnts) > 1:
            cnt = cnts[0]
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            #print cX
            cY = int(M["m01"] / M["m00"])
            #print cY
            ballCentroid = tuple(np.rint((cX, cY)).astype(int))
            print ballCentroid
            foundBall = True
            self.ballPixelLoc = list(ballCentroid)
            # print 'Centroid of ball for frame: ' + str(ballCentroid)
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



def main():

    filename = '../UntrackedFiles/clip25.mp4'
    vr = VideoReader(filename)
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
            #cv2.imshow('frame',cv2.resize(frameDiff, (960, 540)))
            #cv2.waitKey(1)

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
