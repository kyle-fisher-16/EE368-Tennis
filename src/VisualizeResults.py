# script to visualize results of ball tracking + court detection
# expects three files as input:
# 1) video from Megan's camera
# 2) video from Kyle's camera
# 3) text/csv file with frame number and x, y, z coordinates of ball

import sys
import numpy as np
import matplotlib.pyplot as plt
from VideoReader import VideoReader
import cv2 as cv
import csv
from mpl_toolkits.mplot3d import Axes3D


def main():

    # get file names
    videoFile1 = sys.argv[1]
    videoFile2 = sys.argv[2]
    dataFile = sys.argv[3]

    # read in video files
    vr1 = VideoReader(videoFile1)
    vr2 = VideoReader(videoFile2)

    # read in tennis court image
    courtTopView = cv.imread('../SharedData/courtTop.png')
    courtTopView = cv.cvtColor(courtTopView, cv.COLOR_BGR2RGB)
    courtHeight, courtWidth, _ = courtTopView.shape
    netSideView = cv.imread('../SharedData/netSide.png')
    netSideView = cv.cvtColor(netSideView, cv.COLOR_BGR2RGB)
    netHeight, netWidth, _ = netSideView.shape

    # read in results data
    with open(dataFile, 'rb') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=',')
        numFramesData = sum(1 for _ in dataReader)
    csvfile.close()
    ballFound = np.zeros((numFramesData, 1))
    positionData = np.zeros((numFramesData, 3))
    velocityData = np.zeros((numFramesData, 3))
    pixelsKyle = np.zeros((numFramesData, 2))
    pixelsMegan = np.zeros((numFramesData, 2))
    with open(dataFile, 'rb') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=',')
        for row in dataReader:
            frameNumber = int(row[0]) - 1
            found = int(row[1])
            if found:
                xPos = float(row[2])
                yPos = float(row[3])
                zPos = float(row[4])
                xVel = float(row[5])
                yVel = float(row[6])
                zVel = float(row[7])
                xPixelKyle = int(row[8])
                yPixelKyle = int(row[9])
                xPixelMegan = int(row[10])
                yPixelMegan = int(row[11])
                ballFound[frameNumber] = found
                positionData[frameNumber, :] = [xPos, yPos, zPos]
                velocityData[frameNumber, :] = [xVel, yVel, zVel]
                pixelsKyle[frameNumber, :] = [xPixelKyle, yPixelKyle]
                pixelsMegan[frameNumber, :] = [xPixelMegan, yPixelMegan]
    csvfile.close()

    # parameters for results file
    msToMPH = 2.23694
    xPosMin = -7
    xPosMax = 7
    yPosMin = 0
    yPosMax = 3
    zPosMin = -21
    zPosMax = 21

    # parameters for court graphic
    xCourtMin = -8.25
    xCourtMax = 8.25
    xCourtWidth = abs(xCourtMin) + abs(xCourtMax)
    yCourtMin = -14.75
    yCourtMax = 14.75
    yCourtLength = abs(yCourtMin) + abs(yCourtMax)

    # get first frame
    currentFrame = 0
    vr1.setNextFrame(currentFrame)
    ret1, frame1, = vr1.readFrame()
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    frame1Height = vr1.height
    frame1Width = vr1.width
    numFrames1 = int(vr1.numFrames)
    print "Frames in video 1: " + str(numFrames1)
    vr2.setNextFrame(currentFrame)
    ret2, frame2, = vr2.readFrame()
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
    frame2Height = vr2.height
    frame2Width = vr2.width
    numFrames2 = int(vr2.numFrames)
    print "Frames in video 2: " + str(numFrames2)
    print "Frames in data: " + str(numFramesData)
    frameOffset = int(numFrames1 - numFramesData - 5)
    numFrames = int(min(min(numFrames1, numFrames2), numFramesData))

    # compute velocity in MPH
    if ballFound[currentFrame]:
        [xv, yv, zv] = velocityData[currentFrame, :]
        velMPH = np.sqrt((xv * msToMPH) ** 2 + (yv * msToMPH) ** 2 + (zv * msToMPH) ** 2)
        print 'Velocity: ' + str(velMPH) + ' mph'

    # corners of court
    corners1 = [(1161, 431), (1758, 479), (1368, 978), (76, 716)]
    corners2 = [(122, 456), (752, 446), (1754, 817), (319, 1030)]

    # create figure and show first frame
    fig = plt.figure(figsize=(12, 8))
    #fig.tight_layout()
    plt.ion()
    ax1 = fig.add_subplot(2,2,1)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.set_title("Megan's Camera")
    im1 = ax1.imshow(frame1)
    for x, y in corners1:
        ax1.scatter(x, y, s=16, c='red', marker='o')
    ax2 = fig.add_subplot(2,2,2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.set_title("Kyle's Camera")
    im2 = ax2.imshow(frame2)
    for x, y in corners2:
        ax2.scatter(x, y, s=16, c='red', marker='o')

    # top view of court
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    im3 = ax3.imshow(courtTopView)
    #speedText = 'Speed: ' + str(round(velMPH, 2)) + ' mph'
    #ax3.text(4, 4, speedText, fontsize=12)

    # 3D scatterplot
    ax4 = fig.add_subplot(2,2,4, projection = '3d')
    ax4.set_xlim([zPosMin, zPosMax])
    ax4.set_xticks(np.arange(-20, 21, 10))
    ax4.set_xlabel("Court Length (m)")
    ax4.set_ylim([xPosMin, xPosMax])
    ax4.set_yticks(np.arange(-6, 7, 3))
    ax4.set_ylabel("Court Width (m)")
    ax4.set_zlim([yPosMin, yPosMax])
    ax4.set_zticks(np.arange(0, 4, 1))
    ax4.set_zlabel("Ball Height (m)")
    ax4.set_aspect('equal')

    plt.show()

    # output video file
    imFilename = '../UntrackedFiles/imageOutput/image' + str(currentFrame) + '.png'
    plt.savefig(imFilename)
    # vidFrame = cv.imread(imFilename)
    # vidFrameHeight, vidFrameWidth, _ = vidFrame.shape
    # outFilename = '../UntrackedFiles/testVideo.mp4'
    # fourcc = cv.VideoWriter_fourcc(*'mp4v');
    # outputVideo = cv.VideoWriter(outFilename,fourcc, 60, (vidFrameWidth,vidFrameHeight));
    # outputVideo.write(vidFrame)

    # update plots in real-time
    for f in range(1, numFrames):
        currentFrame = f
        vr1.setNextFrame(currentFrame)
        ret1, frame1, = vr1.readFrame()
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        vr2.setNextFrame(currentFrame)
        ret2, frame2, = vr2.readFrame()
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)

        # compute velocity in MPH
        if ballFound[f]:
            [xv, yv, zv] = velocityData[f, :]
            velMPH = np.sqrt((xv * msToMPH) ** 2 + (yv * msToMPH) ** 2 + (zv * msToMPH) ** 2)
            #print 'Velocity: ' + str(velMPH) + ' mph'
            speedText = 'Speed: ' + str(round(velMPH, 2)) + ' mph'
            ax3.set_title(speedText)
            #ax3.text(4, 4, speedText, fontsize=12)

        # Megan's camera
        im1.set_data(frame1)
        if ballFound[f]:  # plot pixel coordinates of ball
            [x1, y1] = pixelsMegan[f,:]
            ax1.scatter(x1, y1, s=1, c='pink', marker='x')

        # Kyle's camera
        im2.set_data(frame2)
        if ballFound[f]:
            [x2, y2] = pixelsKyle[f,:]
            ax2.scatter(x2, y2, s=1, c='pink', marker='x')

        # court graphic
        if ballFound[f]:
            [x, y, z] = positionData[f, :]
            if x > xCourtMin and x < xCourtMax and z > yCourtMin and z < yCourtMax:
                xc = int(round(((x - xCourtMin) / (2 * xCourtMax)) * courtWidth))
                yc = int(round(((z - yCourtMin) / (2 * yCourtMax)) * courtHeight))
                ax3.scatter(xc, yc, s=2, c='pink', marker='o')


        # 3D scatterplot
        if ballFound[f]:
            [x, y, z] = positionData[f, :]
            ax4.scatter(z, x, y, s=2, c='blue', marker='o')

        # update plots

        plt.show()
        imFilename = '../UntrackedFiles/imageOutput/image' + str(f) + '.png'
        fig.set_size_inches(12, 8)
        plt.savefig(imFilename, dpi=100)
        #vidFrame = cv.imread(imFilename)
        #outputVideo.write(vidFrame)
        plt.pause(0.00001)

    # close video readers
    vr1.close()
    vr2.close()
    outputVideo.release()




if __name__ == '__main__':
    if (len(sys.argv) != 4):
        print 'Usage: VisualizeResults.py <video-file> <video-file> <data-file>'
        sys.exit(0)
    main()
