import cv2
import numpy as np

# def FindCourtCorners(frame, file_output=0):
# input: frame -- a numpy array representing the frame in RGB
#        file_output -- flag to produce debugging output at:
#                        ../UntrackedFiles/out/*.png
#                        For this to work, please create
#                        this "out/" directory first.
# output:
#        (success, corners)
#         success -- boolean flag indicating success
#         corners -- 4x2 array containing the coordinates of corners:
#                    [back left; back right; front right; front left;]

# Example:
#   from FindCourtCorners import FindCourtCorners
#   frame = cv2.imread( "../SharedData/FindCourtTest1.png");
#   print FindCourtCorners(frame,1);


# The algorithm works by taking a rectangular crop of the court's
# color (with assumption that it's in the center region of the frame).
# The dominant color of the court is found using histogram of this crop.
# Then, we extract a mask corresponding the the court's shape using
# thresholding, morphological closure, and contour-finding.

# Next, edge-detection is used to find the edges of the court. A hough
# transform is applied to the edges mask, and the dominant lines
# are intersected together. We expect 4 clusters of intersections.
# The clusters are projected to points using dilation and then
# finding the centroid of the dilated blobs. The largest 4 blobs
# are assumed to be the dominant clusters, corresponding to
# actual court corners. Then, the corners are sorted by spatial
# coordinates with ordering [back left, back right, front right,
# front left] as perceived from the camera.

# You can see intermediate output by createing this directory:
# mkdir ../UntrackedFiles/out/
# Then, pass file_output=1 into GetCourtFeaturePoints



class CourtFinder(object):
    def __init__(self):
        self.corners_sort = []
        self.found_corners = False

    # Intersection of rho/theta lines
    def RhoThetaIsect(self, rho1, rho2, theta1, theta2 ):
        term1 = rho2 / np.sin(theta2);
        term2 = rho1 / np.sin(theta1);
        term3 = 1.0/np.tan(theta2) - 1.0/np.tan(theta1);
        x = (term1 - term2) / term3;
        y = (rho1 - x * np.cos(theta1)) / np.sin(theta1);
        return (int(x), int(y));

    # Find dominant color in image using hist. binning
    def GetDominantColor(self, img):
        result = [0,0,0];
        bins = 64;
        bin_w = 256/bins;
        for i in range (0,3):
            hist = cv2.calcHist([img],[i],None,[bins],[0,256]);
            hist_soft = hist[1:-1];
            hist_soft += hist[:-2];
            hist_soft += hist[2:];
            idx = np.argmax(hist_soft);
            result[i] = (idx + 1.5) * bin_w;
        return np.asarray(result);

    # Find the corners of the court
    def FindCourtCorners(self, frame, file_output=0):
        # Get h/w for convenience
        height, width = frame.shape[:2];

        # Take a small window from the center of the image and average its pixels in HSV
        cent_x = int(width / 2);
        cent_y = int(height / 2);
        cent_win_sz = int(width / 20);
        win =  frame[(cent_y - cent_win_sz):(cent_y + cent_win_sz), (cent_x - cent_win_sz):(cent_x + cent_win_sz)];
        if file_output:
            cv2.imwrite( "../UntrackedFiles/out/frame.png", frame);
            cv2.imwrite( "../UntrackedFiles/out/win.jpg", win);

        # Find the biggest region that closely matches the court's average color in HSV space
        win_hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV);
        win_dominant_hsv = self.GetDominantColor(win_hsv);
        sat_thresh = 4;
        hue_thresh = 30;
        val_thresh = 1000;
        lower_sat = win_dominant_hsv - [sat_thresh, hue_thresh, val_thresh];
        upper_sat = win_dominant_hsv + [sat_thresh, hue_thresh, val_thresh];
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv_frame, lower_sat, upper_sat);
        if file_output:
            cv2.imwrite("../UntrackedFiles/out/hsv_mask.png", hsv_mask);

        # Find the biggest region that closely matches the court's average color in RGB space
        win_rgb = win.copy();
        win_dominant_rgb = self.GetDominantColor(win_rgb);
        r_thresh = 40;
        g_thresh = 40;
        b_thresh = 40;
        lower_rgb = win_dominant_rgb - [r_thresh, g_thresh, b_thresh];
        upper_rgb = win_dominant_rgb + [r_thresh, g_thresh, b_thresh];
        rgb_mask = cv2.inRange(frame, lower_rgb, upper_rgb);
        if file_output:
            cv2.imwrite("../UntrackedFiles/out/rgb_mask.png", rgb_mask);
        court_mask = cv2.bitwise_and(rgb_mask, rgb_mask, mask=hsv_mask);
        if file_output:
            cv2.imwrite("../UntrackedFiles/out/court_mask.png", court_mask);

        # Output the court's dominant RGB color for preview
        preview = np.ones((100,100,3)) * np.asarray([win_dominant_rgb[0], win_dominant_rgb[1], win_dominant_rgb[2]]);
        if file_output:
            cv2.imwrite("../UntrackedFiles/out/court_color_rgb.png", preview);

        # Find the largest contour in the court mask. This is assumed to be the court.
        close_sz = int(width / 96);
        dilate_sz = int(width / 150);
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, np.ones((close_sz,close_sz)))
        im2, contours, hier = cv2.findContours(court_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0;
        max_c = None;
        for c in contours:
            c_area = cv2.contourArea(c)
            if c_area > max_area:
                max_area = c_area
                max_c = c;

        # Draw the outline of the court
        court_mask_color = np.zeros(frame.shape);
        court_mask_color = (cv2.drawContours(court_mask_color,[max_c],0,(0, 255, 0),-1));
        court_mask = cv2.inRange(court_mask_color, (0,255,0), (0, 255, 0));
        court_outline = cv2.Canny(court_mask,100,200)

        # Dilate the outline to help out the Hough transform.
        dilate_sz = int(width / 450);
        court_outline = cv2.morphologyEx(court_outline, cv2.MORPH_DILATE, np.ones((dilate_sz,dilate_sz)))
        if file_output:
            cv2.imwrite( "../UntrackedFiles/out/court_outline.jpg", court_outline);


        # Do a Hough transform to find dominant lines.
        frame_lines = frame.copy();
        lines = cv2.HoughLines(court_outline,1,np.pi/180, int(width/8));
        isect_mask = np.zeros(court_outline.shape, dtype = "uint8");

        # Find all interesting intersections among Hough lines
        angle_thresh = 0.1; # pairs of lines with relative angles smaller than this are ignored
        for line1 in lines:
            rho1 = line1[0][0];
            theta1 = line1[0][1] + 0.0005234; # fix div-zero case
            for line2 in lines:
                rho2 = line2[0][0];
                theta2 = line2[0][1] + 0.0005234;  # fix div-zero case
                if (theta1 != theta2):
                    #print rho1, rho2, theta1, theta2
                    isect = self.RhoThetaIsect(rho1, rho2, theta1, theta2);
                    # TODO: handle edge cases of theta1-theta2
                    if (isect[0] >= 0 and isect[0] < width and isect[1] >= 0 and isect[1] < height) and np.abs(theta1-theta2) > angle_thresh:
                      isect_mask[isect[1]][isect[0]] = 1;


        # Draw the Hough lines for debugging
        for line in lines:
            line = np.squeeze(line);
            rho = line[0];
            theta = line[1] + 0.0001; # hack, ensures eventual intersection...
            isect1 = self.RhoThetaIsect(rho, 0, theta, 0.001); # vertical
            isect2 = self.RhoThetaIsect(rho, 0, theta, np.pi/2); # horiz
            isect3 = self.RhoThetaIsect(rho, height, theta, np.pi/2); # horiz
            cv2.line(frame_lines,isect1,isect2,(0,0,255),2);
            if (isect2[0] > isect3[0]):
                cv2.line(frame_lines,isect1,isect2,(0,0,255),2);
            else:
                cv2.line(frame_lines,isect1,isect3,(0,0,255),2);
        if file_output:
            cv2.imwrite("../UntrackedFiles/out/houghlines.png",frame_lines);

        # Find centroids among intersection clusters. These are considered corners.
        dilate_sz = int(width / 50);
        isect_mask = cv2.morphologyEx(isect_mask, cv2.MORPH_DILATE, np.ones((dilate_sz,dilate_sz)));
        dilate_sz = int(width / 35);
        court_mask_dilated = cv2.morphologyEx(court_mask, cv2.MORPH_DILATE, np.ones((dilate_sz,dilate_sz)));
        isect_mask = isect_mask  & court_mask_dilated;
        im2, contours, hier = cv2.findContours(isect_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);
        if file_output:
            cv2.imwrite("../UntrackedFiles/out/isect_dots.png", isect_mask * 255);
        # if there are fewer than 4 contours, we failed to find 4 court corners in the image.
        if (len(contours) < 4):
            # return (False, []);
            self.corners_sort = []
            self.found_corners = False
            return
        # sort the corners by their confidence (in this case, area of the blob of intersections)
        area_idx = np.argsort([-cv2.contourArea(c) for c in contours]);
        area_idx = area_idx[:4]
        corners = np.zeros((4,2), dtype="uint32");
        ct = 0;
        for idx in area_idx:
            M = cv2.moments(contours[idx])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            isect_mask[cy][cx] = 0;
            corners[ct] = [cx, cy]; ct += 1;

        # Sort the corners based on where they are located (clockwise indexing).
        y_sort_idx = np.argsort([c[1] for c in corners]);
        corners_sorted = corners[y_sort_idx, :];
        x_sort_idx = [0,1,2,3];
        if corners_sorted[0,0] > corners_sorted[1,0]:
            x_sort_idx[0] = 1;
            x_sort_idx[1] = 0;
        if corners_sorted[2,0] < corners_sorted[3,0]:
            x_sort_idx[2] = 3;
            x_sort_idx[3] = 2;
        corners_sorted = corners_sorted[x_sort_idx,:];

        # Draw the frame with marked corners (for debugging)
        if file_output:
            frame_marked_corners = frame.copy();
            corner_idx = 0;
            for corner in corners_sorted:
                cx = corner[0];
                cy = corner[1];
                # draw this corner
                draw_length=15;
                cv2.line(frame_marked_corners,(cx - draw_length, cy),(cx + draw_length,cy),(0, 0, 255),2);
                cv2.line(frame_marked_corners,(cx, cy - draw_length),(cx,cy + draw_length),(0, 0, 255),2);
                font = cv2.FONT_HERSHEY_SIMPLEX;
                bottomLeftCornerOfText = (cx + draw_length,cy - draw_length);
                fontScale = 1;
                fontColor = (0,0,255);
                lineType = 2;
                cv2.putText(frame_marked_corners,str(corner_idx),
                          bottomLeftCornerOfText,
                          font,
                          fontScale,
                          fontColor,
                          lineType);
                corner_idx += 1;
            cv2.imwrite("../UntrackedFiles/out/frame_marked_corners.png", frame_marked_corners);
        # return (True, corners_sorted);
        # set self.corners_sort and self.found_corners
        self.corners_sort = corners_sorted
        self.found_corners = True

    def drawCornersOnFrame(self, frame):
        frame_marked_corners = frame.copy();
        corner_idx = 0;
        for corner in self.corners_sort:
            cx = corner[0];
            cy = corner[1];
            # draw this corner
            draw_length=15;
            cv2.line(frame_marked_corners,(cx - draw_length, cy),(cx + draw_length,cy),(0, 0, 255),2);
            cv2.line(frame_marked_corners,(cx, cy - draw_length),(cx,cy + draw_length),(0, 0, 255),2);
            font = cv2.FONT_HERSHEY_SIMPLEX;
            bottomLeftCornerOfText = (cx + draw_length,cy - draw_length);
            fontScale = 1;
            fontColor = (0,0,255);
            lineType = 2;
            cv2.putText(frame_marked_corners,str(corner_idx),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType);
            corner_idx += 1;
        return frame_marked_corners
