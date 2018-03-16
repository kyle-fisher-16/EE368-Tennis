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
        # self.ballColHsv = np.array([int(44), int(115), int(240)])
        self.ballColHsv = np.array([int(44), 127, int(230)])
        # far away balls have very small saturation levels
        satThresh = 127;
        valThresh = 25;
        hueThresh = 40;
        self.lowerCol = np.array([self.ballColHsv[0]-hueThresh, 0, 220])
        self.upperCol = np.array([self.ballColHsv[0]+hueThresh, 255, 255])
        # centroid of found ball in motion for tracking
        self.ballPixelLoc = [0,0]
        # maybe we should try like kalman filtering this or something?

    # returns the binary frame difference between frames 1 and 2
    # use to locate the ball initially where frame2 for background subtraction
    # set withFilt to False if don't want to perform opening to get rid of noise
        # if ball is far away it is very small and
        # may be smaller than noise in foreground
    def hsvDiff(self, frame1, frame2, withFilt=True):
        hsvFrame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsvFrame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        # hsv narrow range:
        hue = self.ballColHsv[0]
        hueThresh = 15
        lowerCol = np.array([hue-hueThresh, self.lowerCol[1], self.lowerCol[2]])
        upperCol = np.array([hue+hueThresh, self.upperCol[1], self.upperCol[2]])
        mask1 = cv2.inRange(hsvFrame1, lowerCol, upperCol)
        mask2 = cv2.inRange(hsvFrame2, lowerCol, upperCol)
        # remember that these are uints, so no negative values
        # xor to get 2 distinct balls, then and with original to only get first ball
        frameDiff = mask1 ^ mask2
        frameBall1 = frameDiff & mask1
        if withFilt:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            frameBall1Filt = cv2.morphologyEx(frameBall1, cv2.MORPH_OPEN, se)
            return frameBall1Filt
        else:
            return frameBall1

    def rgbDiff(self, frame1, frame2):
        frameDiff = frame1.astype(np.int16) - frame2.astype(np.int16)
        frameDiff = abs(frameDiff)
        frameDiff = frameDiff.astype(np.uint8)
        # threshold difference
        frameDiffGray = cv2.cvtColor(frameDiff, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(frameDiffGray, 50, 255)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
        maskFilt = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        maskFilt = cv2.morphologyEx(maskFilt, cv2.MORPH_DILATE, se)
        # add filter by eccenetricity to get rid of too big and components etc
        ___notsure, contours, hier = cv2.findContours(maskFilt,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        maskEcc = np.zeros(maskFilt.shape, dtype="uint8");
        for c in contours:
            cArea = cv2.contourArea(c)
            if len(c) < 5: # too small to measure eccenetricity
                cv2.drawContours(maskEcc,[c],0,255,-1)
                continue;
            c_ellipse = cv2.fitEllipse(c)
            rect = c_ellipse[1];
            ecc = np.max([rect[0]/(rect[1]+0.001), rect[1]/(rect[0]+0.001)]);
            if ecc < 2 and cArea < 200:
                cv2.drawContours(maskEcc,[c],0,255,-1)

        colorFiltMask = self.hsvFilt(frame1, False)
        maskFilt = colorFiltMask & maskEcc
        return maskFilt

    # returns the binary frame filtered in HSV space where filtered for ball color
    def hsvFilt(self, frame, withFilt=True):
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ballFrame = cv2.inRange(hsvFrame, self.lowerCol, self.upperCol)
        if withFilt:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            frameBallFilt = cv2.morphologyEx(ballFrame, cv2.MORPH_OPEN, se)
            return frameBallFilt
        else:
            return ballFrame

    # GetCornernessMask(frame, thresh, dilateSz)
    #    frame: RGB image
    #    thresh: lower threshold for corner to be registered
    #            set to 0.0 to register everything
    #            set to 1.0 to register nothing
    #    dilateSz: size to dilate the corners so they make a useful mask
    def GetCornernessMask(self, frame1, frame2, thresh=0.01, dilateSz=6):
        frameDiff = frame1.astype(np.int16) - frame2.astype(np.int16)
        frameDiff = abs(frameDiff)
        frameDiff = frameDiff.astype(np.uint8)
        # threshold difference
        gray = cv2.cvtColor(frameDiff, cv2.COLOR_BGR2GRAY)
        colorFiltMask = self.hsvFilt(frame1, False)
        gray = gray & colorFiltMask
        # gray = cv2.cvtColor(frameIn,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray);
        dst = cv2.cornerHarris(gray,2,3,0.04);
        dst = cv2.dilate(dst,None);
        cornersMask = np.zeros(gray.shape, dtype="uint8");
        cornersMask[dst>thresh*dst.max()] = 255;
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilateSz,dilateSz))
        cornersMask = cv2.morphologyEx(cornersMask, cv2.MORPH_DILATE, se)
        return cornersMask;

    def aveMask(self, mask1, mask2, mask3):
        sumMask = np.zeros(mask1.shape, dtype="int32")
        sumMask = mask1.astype(np.int32) + mask2.astype(np.int32) + mask3.astype(np.int32)
        aveMask = sumMask/3;
        aveMask = aveMask.astype(np.uint8)
        mask = cv2.inRange(aveMask, int(2*255/3), 255)
        return mask

    # calculate the centroid of a ball where frame_mask is a binary frame
    # this only works if the ball is the largest foreground object
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
