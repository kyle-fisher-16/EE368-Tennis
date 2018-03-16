# for useability interface/ user input to quit display
import sys
import csv

import cv2
import numpy as np

from VideoReader import VideoReader
from FindBall import BallFinder
from FindCourtCorners import CourtFinder
from Camera import Camera, IntersectRays
from kalmanFilter import KalmanFilter

# Gets list of pixel coordinates of ball candidate positions
def getBallCandidateRays(bf, cam, frame1, frame2):
    frameDiff1 = bf.hsvDiff(frame1, frame2)
    frameDiff2 = bf.rgbDiff(frame1, frame2)
    frameCornerMask = bf.GetCornernessMask(frame1, frame2)
    mask = bf.aveMask(frameDiff1, frameDiff2, frameCornerMask)
    im2, contours, hier = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates = [];
    for c in contours:
        center = cv2.minAreaRect(c)[0];
        ray = cam.GetRay(center);
        candidates.append(ray);
    return (np.dstack([mask,mask,mask]), candidates)
def main():

    # Pretty videos:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v');
    kyleOutputVideo = cv2.VideoWriter('../UntrackedFiles/BallOutputKyle.mp4',fourcc, 60.0, (1920,1080));
    meganOutputVideo = cv2.VideoWriter('../UntrackedFiles/BallOutputMegan.mp4',fourcc, 60.0, (1920,1080));
    meganFilename = '../UntrackedFiles/stereoClip5_Megan.mov'
    kyleFilename = '../UntrackedFiles/stereoClip5_Kyle.mov'

    vrMegan1 = VideoReader(meganFilename)
    vrMegan2 = VideoReader(meganFilename)
    vrKyle1 = VideoReader(kyleFilename)
    vrKyle2 = VideoReader(kyleFilename)
    numFrameForward = 5
    vrMegan2.setNextFrame(numFrameForward)
    vrKyle2.setNextFrame(numFrameForward)

    # find court corners:
    cfKyle = CourtFinder()
    cfMegan = CourtFinder()
    numFrames = int(vrMegan1.getNumFrames())
    vrMegan1.setNextFrame(int(numFrames/2))
    ret, frame = vrMegan1.readFrame()
    cfMegan.FindCourtCorners(frame, 0)
    vrKyle1.setNextFrame(int(numFrames/2))
    ret, frame = vrKyle1.readFrame()
    cfKyle.FindCourtCorners(frame, 0)
    if (not cfMegan.found_corners) or not(cfKyle.found_corners):
        print "Couldn't find the court. Exiting."
        return

    # reset frame index to beginning
    vrMegan1.setNextFrame(0)
    vrKyle1.setNextFrame(0)
    frameNum = 1;

    meganCam = Camera("megan", cfMegan.corners_sort);
    kyleCam = Camera("kyle", cfKyle.corners_sort);

    # make a ball finder
    bf = BallFinder()
    kf = KalmanFilter(vrMegan1.framerate)

    csvData = []
    while(True):
        ret1, kyleFrame1 = vrKyle1.readFrame()
        ret2, kyleFrame2, = vrKyle2.readFrame()
        ret3, meganFrame1 = vrMegan1.readFrame()
        ret4, meganFrame2, = vrMegan2.readFrame()
        if not(ret1) or not(ret2) or not (ret3) or not (ret4):
            print 'Ending after', frameNum-1, 'frames.'
            break;

        kyleMask, kyleRays = getBallCandidateRays(bf, kyleCam, kyleFrame1, kyleFrame2);
        meganMask, meganRays = getBallCandidateRays(bf, meganCam, meganFrame1, meganFrame2);

        # check all candidate rays for candidate balls
        minDist = 1000000; # TODO set inf
        ballPt = [];
        # all ball points and distances
        threshDist = 0.2;   # rays must be within .1 m of each other for intersect
        ballCandidates = [];
        candidateCertainty = [];
        for kyleRay in kyleRays:
            for meganRay in meganRays:
                pt, dist, _D, _E = IntersectRays(kyleRay, meganRay)
                if dist < threshDist and pt[1] < 3.5:
                    # don't include candidates clearly not valid intersect points
                    # also don't include candidates that are clearly too high to be the ball
                    courtBuffer = 2
                    if pt[0]< Camera.HALF_COURT_X+courtBuffer and pt[0]> -Camera.HALF_COURT_X-courtBuffer:
                        if pt[2] < Camera.HALF_COURT_Z+0.6:# and pt[2] > -Camera.HALF_COURT_Z - 0:
                            ballCandidates.append(pt)
                            candidateCertainty.append(dist)
                    if dist < minDist:
                        minDist = dist;
                        ballPt = pt;
        kf.processMeas(ballCandidates,candidateCertainty)

        # ========== CSV and VIDEO output ===========
        csvTuple = list();
        csvTuple.append(frameNum);
        if np.linalg.norm(kf.sigma_k,'fro') < 100: # valid result
            # Format the tuple for successful reading
            csvTuple.append(1);
            posVel = np.reshape(kf.mu_k, (1,6))[0];
            posVel = np.round(posVel, 3);
            for val in posVel:
                csvTuple.append(val);
            for val in kyleCam.ConvertWorldToImagePosition(posVel[0:3]):
                csvTuple.append(val);
            for val in meganCam.ConvertWorldToImagePosition(posVel[0:3]):
                csvTuple.append(val);
            # Videos
            kyleOutFrame = kyleFrame1.copy();
            bf.ballPixelLoc = kyleCam.ConvertWorldToImagePosition(posVel[0:3]);
            kyleOutFrame = cfKyle.drawCornersOnFrame(kyleOutFrame)
            kyleOutFrame = bf.drawBallOnFrame(kyleOutFrame);
            kyleOutFrame |= kyleMask;
            kyleOutputVideo.write(kyleOutFrame);
            meganOutFrame = meganFrame1.copy();
            meganOutFrame |= meganMask;
            bf.ballPixelLoc = meganCam.ConvertWorldToImagePosition(posVel[0:3]);
            meganOutFrame = cfMegan.drawCornersOnFrame(meganOutFrame)
            meganOutFrame = bf.drawBallOnFrame(meganOutFrame);
            meganOutputVideo.write(meganOutFrame);
        else:
            # Format the tuple for unsuccessful reading
            csvTuple.append(0);
        csvData.append(list(csvTuple))
        print csvTuple
        frameNum += 1;
        # <END WHILE LOOP>

    with open('../UntrackedFiles/ballEst.csv', 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerows(csvData)

    vrKyle1.close()
    vrKyle2.close()
    vrMegan1.close()
    vrMegan2.close()
    kyleOutputVideo.release();
    meganOutputVideo.release();

if __name__ == '__main__':
        main()
