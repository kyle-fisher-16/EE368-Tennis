# script to take a video file and recognize balls
# for now only ball candidates, but utlimately only moving balls

# import video reader class. File in same folder (src)
from VideoReader import VideoReader
# still need opencv for other image processing stuff
import cv2 as cv

# need numpy
import numpy as np

# for useability interface/ user input to quit display
import sys

def main():
    vr = VideoReader('../UntrackedFiles/clip25.mp4')



    # background subtraction:
    # maybe don't get immediate frame so more of a difference in ball location
        # and ball doesn't get totally subtracted?
    num_frames = int(vr.getNumFrames())
    #print 'Number of frames: ' + str(num_frames)
    # frame_id = vr.getNextFrameIdx()
    #frame_id = 90
    for frame_id in range(0, num_frames):
        vr.setNextFrame(frame_id)
        ret, frame = vr.readFrame()
        #cv.imshow('original frame '+str(frame_id), cv.resize(frame, (960, 540)))
        # only process green channel since balls are green and will hopefully
            # show most distinctly in green channel
        # threshold(input image, threshold, max value, type of thresholding)
        thresh, frame_bw = cv.threshold(frame[:,:,1], 220, 255, cv.THRESH_BINARY)
        #cv.imshow('binary frame ' + str(frame_id), cv.resize(frame_bw, (960, 540)))
        c = cv.waitKey(1) & 0xFF
        if frame_id+1 < num_frames:
            vr.setNextFrame(frame_id+1)  # why +5?
        ret, frame2 = vr.readFrame()
        thresh, frame2_bw = cv.threshold(frame2[:,:,1], 220, 255, cv.THRESH_BINARY)
        # performing subtraction on uints will zero out second frame to get only ball from first frame
        # we can then get ball starting location
        # maybe try with dilated background to subract more of background?
        se = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        frame2_bw_dilate = cv.dilate(frame2_bw, se)
        frame_diff = frame_bw - frame2_bw   # maybe try bitwise xor--> will give 2 balls, one from each frame
        # perform opening to get rid of noise/ non ball pixels
        frame_diff = cv.morphologyEx(frame_diff, cv.MORPH_OPEN, se)
        cv.imwrite('../UntrackedFiles/frame_diff.jpg', frame_diff)
        #cv.imshow('frame diff ' + str(frame_id), cv.resize(frame_diff, (960, 540)))

        # find connected components to extract the location of the ball(s)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 8
        # Perform the operation
        output = cv.connectedComponentsWithStats(frame_diff, connectivity)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
        #largest_label = np.argmax(stats[1:, cv.CC_STAT_AREA])
        # background is largest component with label 0
        # ball should be next largest component with label 1
        if num_labels > 1:
            ball_only = np.zeros_like(frame_diff)
            ball_label = 1
            ball_only[labels == ball_label] = 255
            #cv.imshow('ball only ' + str(frame_id), cv.resize(ball_only, (960, 540)))
            #print 'Number of connected components found: ' + str(num_labels)
            #print 'Label of largest component: ' + str(largest_label)
            ball_centroid = centroids[ball_label]
            print 'Centroid of ball for frame ' + str(frame_id) + ': ' + str(ball_centroid)

            # put circle on top of ball in original frame
            # make the circle orange
            #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            ball_frame = cv.circle(frame, tuple(np.rint(ball_centroid).astype(int)), 10, (180,105,255), -1)
            cv.imshow('ball in frame ' + str(frame_id), cv.resize(ball_frame, (960, 540)))
            cv.waitKey(200)
        else:
            print 'No ball found in frame ' + str(frame_id)


        #c = cv.waitKey(1) & 0xFF



        # vr.playVideo()

        # quit sequence:
        #print "press q enter to quit "
        #done = False
        #while(not(done)):
            #c = sys.stdin.read(1)
            #if c == 'q':
            #    done = True

    vr.close()

if __name__ == '__main__':
    main()
