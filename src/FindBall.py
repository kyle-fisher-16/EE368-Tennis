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
        ball_col_hsv = np.array([int(44), int(115), int(250)])
        # far away balls have very small saturation levels
        sat_thresh = 75;
        val_thresh = 5;
        hue_thresh = 20;
        lower_col = ball_col_hsv - [hue_thresh, sat_thresh, val_thresh]
        upper_col = ball_col_hsv + [hue_thresh, sat_thresh, val_thresh]
        # centroid of found ball in motion for tracking
        ball_pixel_loc = []


def main():
    # TODO: code restructure
    # find better spot for these...
    # hsv in opencv: h [0,179], s [0,255], v [0,255]
    ball_col_hsv = np.array([int(44), int(115), int(250)])
    # far away balls have very small saturation levels
    sat_thresh = 75;
    val_thresh = 5;
    hue_thresh = 20;
    lower_col = ball_col_hsv - [hue_thresh, sat_thresh, val_thresh]
    upper_col = ball_col_hsv + [hue_thresh, sat_thresh, val_thresh]

    vr = VideoReader('../UntrackedFiles/clip25.mp4')

    # playing with HSV space
    vr.setNextFrame(90)
    frame_id = vr.getNextFrameIdx()
    ret, frame1 = vr.readFrame()
    # vr.setNextFrame(frame_id+2)
    vr.setNextFrame(88)
    ret, frame2 = vr.readFrame()
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_frame1, lower_col, upper_col)
    mask2 = cv2.inRange(hsv_frame2, lower_col, upper_col)
    frame_diff = mask1 - mask2
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    frame_diff_filt = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, se)
    # cv2.imwrite('../UntrackedFiles/hsv_mask.jpg', mask)
    cv2.imshow('hsv frame', cv2.resize(frame_diff_filt, (960, 540)))
    cv2.waitKey(1)

    # quit sequence:
    print "press q enter to quit "
    done = False
    while(not(done)):
        c = sys.stdin.read(1)
        if c == 'q':
           done = True

    vr.close()
    return




    # background subtraction:
    # maybe don't get immediate frame so more of a difference in ball location
        # and ball doesn't get totally subtracted?
    num_frames = int(vr.getNumFrames())
    #print 'Number of frames: ' + str(num_frames)
    # frame_id = vr.getNextFrameIdx()
    #frame_id = 90
    for frame_id in range(0, num_frames-1):
    #frame_id = 10
        vr.setNextFrame(frame_id)
        ret, frame = vr.readFrame()
        #cv2.imshow('original frame '+str(frame_id), cv2.resize(frame, (960, 540)))
        # only process green channel since balls are green and will hopefully
            # show most distinctly in green channel
        # threshold(input image, threshold, max value, type of thresholding)
        thresh, frame_bw = cv2.threshold(frame[:,:,1], 220, 1, cv2.THRESH_BINARY)
        # remove white pixels from tresholded values
        thresh, frame_red = cv2.threshold(frame[:,:,2], 240, 1, cv2.THRESH_BINARY_INV)
        thresh, frame_blue = cv2.threshold(frame[:,:,0], 230, 1, cv2.THRESH_BINARY_INV)
        frame_bw_ball = np.logical_and(frame_bw, np.logical_and(frame_red, frame_blue))
        frame_bw_ball = frame_bw_ball.astype(float) * 255
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        frame_bw_ball = cv2.morphologyEx(frame_bw_ball, cv2.MORPH_CLOSE, se)
        #cv2.imshow('frame bw ball', cv2.resize(frame_bw_ball, (960, 540)))

        #cv2.imshow('binary frame ' + str(frame_id), cv2.resize(frame_bw, (960, 540)))
        c = cv2.waitKey(1) & 0xFF
        if frame_id+5 < num_frames:
            vr.setNextFrame(frame_id+5)
        ret, frame2 = vr.readFrame()
        thresh, frame2_bw = cv2.threshold(frame2[:,:,1], 220, 1, cv2.THRESH_BINARY)
        thresh, frame_red = cv2.threshold(frame[:,:,2], 240, 1, cv2.THRESH_BINARY_INV)
        thresh, frame_blue = cv2.threshold(frame[:,:,0], 230, 1, cv2.THRESH_BINARY_INV)
        frame2_bw_ball = np.logical_and(frame2_bw, np.logical_and(frame_red, frame_blue))
        frame2_bw_ball = frame2_bw_ball.astype(float) * 255
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        frame2_bw_ball = cv2.morphologyEx(frame2_bw_ball, cv2.MORPH_CLOSE, se)
        #cv2.imshow('frame2 bw ball', cv2.resize(frame2_bw_ball, (960, 540)))


        # performing subtraction on uints will zero out second frame to get only ball from first frame
        # we can then get ball starting location
        # maybe try with dilated background to subract more of background?
        #se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        #frame2_bw_dilate = cv2.dilate(frame2_bw, se)
        frame_diff = frame_bw_ball - frame2_bw_ball   # maybe try bitwise xor--> will give 2 balls, one from each frame
        # perform opening to get rid of noise/ non ball pixels
        frame_diff = cv2.morphologyEx(frame_diff, cv2.MORPH_OPEN, se)
        frame_diff = frame_diff.astype(np.uint8)
        #cv2.imwrite('../UntrackedFiles/frame_diff.jpg', frame_diff)
        cv2.imshow('frame diff ' + str(frame_id), cv2.resize(frame_diff, (960, 540)))
        #c = cv2.waitKey(1) & 0xFF

        # find connected components to extract the location of the ball(s)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4
        # Perform the operation
        output = cv2.connectedComponentsWithStats(frame_diff, connectivity)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
        #largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # background is largest component with label 0
        # ball should be next largest component with label 1

        cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 1:
        #if num_labels > 1:
            #ball_only = np.zeros_like(frame_diff)
            #ball_label = 2
            #ball_only[labels == ball_label] = 255
            #cv2.imshow('ball only ' + str(frame_id), cv2.resize(ball_only, (960, 540)))
            #print 'Number of connected components found: ' + str(num_labels)
            #print 'Label of largest component: ' + str(largest_label)

            #cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = cnts[0]
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            #print cX
            cY = int(M["m01"] / M["m00"])
            #print cY
            ball_centroid = tuple(np.rint((cX, cY)).astype(int))

                # # draw the contour and center of the shape on the image
                # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                # cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                # cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # # show the image
                # cv2.imshow("Image", image)


            #ball_centroid = centroids[ball_label]
            print 'Centroid of ball for frame ' + str(frame_id) + ': ' + str(ball_centroid)

            # put circle on top of ball in original frame
            # make the circle orange
            #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            ball_frame = cv2.circle(frame, ball_centroid, 10, (180,105,255), -1)
            cv2.imshow('ball in frame ' + str(frame_id), cv2.resize(ball_frame, (960, 540)))
            cv2.waitKey(10)
            #c = cv2.waitKey(1) & 0xFF
        else:
            print 'No ball found in frame ' + str(frame_id)




            #c = cv2.waitKey(1) & 0xFF



            # vr.playVideo()

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
