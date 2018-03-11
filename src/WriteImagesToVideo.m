% EE 386 - Digital Image Processing
% Winter 2017-18
% Megan Fazio, Kyle Fisher, Tori Fujinami
% Final Project
% Create viz video from image files

clear, close all

% create video file
outputVideo = VideoWriter('testVideo.mp4', 'MPEG-4');
outputVideo.FrameRate = 60;
open(outputVideo);

% read in images
numRange = 0:200;
numImages = length(numRange);
images = cell(1, numImages);
for i = 1:numImages
    imgNum = numRange(i);
    img = imread(sprintf('/Users/Megan/Documents/EE 368 Project/UntrackedFiles/imageOutput/image%d.png', imgNum));
    writeVideo(outputVideo,img)
end

close(outputVideo);