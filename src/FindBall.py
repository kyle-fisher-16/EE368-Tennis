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



    # background subtraction:
    # maybe don't get immediate frame so more of a difference in ball location
        # and ball doesn't get totally subtracted?
    num_frames = vr.getNumFrames()
    # frame_id = vr.getNextFrameIdx()
    frame_id = 90
    vr.setNextFrame(frame_id)
    ret, frame = vr.readFrame()
    # only process green channel since balls are green and will hopefully
        # show most distinctly in green channel
    # threshold(input image, threshold, max value, type of thresholding)
    thresh, frame_bw = cv.threshold(frame[:,:,1], 220, 255, cv.THRESH_BINARY)
    cv.imshow('binary frame', frame_bw)
    c = cv.waitKey(1) & 0xFF
    if frame_id+5 < num_frames:
        vr.setNextFrame(frame_id+5)
    ret, frame2 = vr.readFrame()
    thresh, frame2_bw = cv.threshold(frame2[:,:,1], 220, 255, cv.THRESH_BINARY)
    # performing subtraction on uints will zero out second frame to get only ball from first frame
    # we can then get ball starting location
    # maybe try with dilated background to subract more of background?
    se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    frame2_bw_dilate = cv.dilate(frame2_bw, se)
    frame_diff = frame_bw - frame2_bw   # maybe try bitwise xor--> will give 2 balls, one from each frame
    # perform opening to get rid of noise/ non ball pixels
    # frame_diff = cv.morphologyEx(frame_diff, cv.MORPH_OPEN, se)
    # cv.imwrite('../UntrackedFiles/frame_diff.jpg', frame_diff)
    cv.imshow('frame diff', frame_diff)
    c = cv.waitKey(1) & 0xFF



    # vr.playVideo()

    # quit sequence:
    print "press q enter to quit "
    done = False
    while(not(done)):
        c = sys.stdin.read(1)
        if c == 'q':
            done = True

    vr.close()

if __name__ == '__main__':
    main()
