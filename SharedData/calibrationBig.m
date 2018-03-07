% Auto-generated by cameraCalibrator app on 06-Mar-2018
%-------------------------------------------------------


% Define images to process
imageFileNames = {'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0388.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0389.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0390.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0391.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0392.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0393.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0394.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0395.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0396.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0397.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0398.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0399.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0400.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0401.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0402.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0403.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0404.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0405.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0406.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0407.jpg',...
    'C:\Users\Megan\Dropbox\Stanford MBA-MSEE\Winter 17-18\EE 368 - Digital Image Processing\Final Project\CalibrationPhotos\BigTarget\IMG_0408.jpg',...
    };

% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates of the corners of the squares
squareSize = 29;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', true, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')