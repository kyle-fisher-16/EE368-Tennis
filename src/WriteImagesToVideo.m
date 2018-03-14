% EE 386 - Digital Image Processing
% Winter 2017-18
% Megan Fazio, Kyle Fisher, Tori Fujinami
% Final Project
% Create viz video from image files

clear, close all

frameRates = [10, 30, 60];
clipNumber = 12;
numRange = 0:162;

for f = 1:length(frameRates)
    
    frameRate = frameRates(f);
    fprintf('Creating video file at %d fps...\n', frameRate);

    % create video file
    outputVideo = VideoWriter(sprintf('/Users/Megan/Documents/EE 368 Project/UntrackedFiles/demoClip%d_%dfps.mp4', clipNumber, frameRate), 'MPEG-4');
    outputVideo.FrameRate = frameRate;
    open(outputVideo);

    % read in images
    numImages = length(numRange);
    images = cell(1, numImages);
    for i = 1:numImages
        imgNum = numRange(i);
        img = imread(sprintf('/Users/Megan/Documents/EE 368 Project/UntrackedFiles/imageOutput/image%d.png', imgNum));
        writeVideo(outputVideo,img)
    end

    close(outputVideo);
end