
import cv2
import numpy as np

class Camera:
  
  CameraMatrix = [];
  DistCoeffs = [];
  Position = [];
  RotationVec = [];
  TranslationVec = [];
  CourtCorners = [];
  Homog = [];

  HALF_COURT_X = 4.115;
  HALF_COURT_Z = 11.885;
  WORLD_POINTS = np.asarray([[-HALF_COURT_X, 0, -HALF_COURT_Z],
                             [ HALF_COURT_X, 0, -HALF_COURT_Z],
                             [ HALF_COURT_X, 0,  HALF_COURT_Z],
                             [-HALF_COURT_X, 0,  HALF_COURT_Z]]);
  

  def __init__(self, cameraName, courtCorners):
    if cameraName == "kyle":
      fx=1994.25368447834;
      fy=1988.65266798629;
      cx=968.573023612607;
      cy=511.585679422200;
      k1=0.0771110325943740;
      k2=-0.0596894545787290;
      p1=0.00178967197419077;
      p2=0.00123017525081653;
    elif cameraName == "megan":
      fx=1981.39204255929;
      fy=1973.70141739089;
      cx=980.523462971786;
      cy=551.217098728122;
      k1=0.0747612507420630;
      k2=-0.0683271738685350;
      p1=0.00240502474003212;
      p2=0.00199735586169493;
    else:
      raise ValueError("cameraName must be 'kyle' or 'megan'!")
      return;

    self.CourtCorners = courtCorners.copy();
    self.CameraMatrix = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]);
    self.DistCoeffs = np.zeros((4,1));#np.asarray([k1, k2, p1, p2]);
  
    # FIND CAMERA POSITION
    imgCoords = np.transpose(courtCorners);
    _, rVec, tVec = cv2.solvePnP(self.WORLD_POINTS.reshape((4,1,3)), courtCorners.reshape((4,1,2)), self.CameraMatrix, self.DistCoeffs,flags=cv2.SOLVEPNP_ITERATIVE);
    self.RotationVec = rVec.copy();
    self.Rotation = cv2.Rodrigues(rVec)[0];
    self.TranslationVec = tVec.copy();
    R_inv = np.transpose(self.Rotation);
    self.Position = - (np.matmul(R_inv,tVec))
    #print self.Position

    # FIND MAPPING FROM CAM TO WORLD
    camPoints = np.zeros((4,2), dtype="float32");
    for i in range(0,4):
      pt = self.GetPinholePoint(self.CourtCorners[i,:]);
      camPoints[i,0] = pt[0]; # U coord
      camPoints[i,1] = pt[1]; # V coord
    worldPoints = self.WORLD_POINTS[:, [0,2]]
    self.Homog = cv2.findHomography(camPoints, worldPoints)[0];

  # Undistort the pixel position and convert it to pinhole coordinates w/ focal length 1
  def GetPinholePoint(self, pt):
    pts = np.zeros((1,1,2));
    pts[0,0,0] = pt[0];
    pts[0,0,1] = pt[1];
    result = cv2.undistortPoints(pts, self.CameraMatrix, self.DistCoeffs);
    xy = np.asarray([result[0,0,0], result[0,0,1]]);
    return xy

  # Convert a point from pixel position to court position
  def ConvertPixelToCourtPosition(self, pt):
    pinholePt = self.GetPinholePoint(pt);
    pt2 = np.asarray([pinholePt[0], pinholePt[1], 1.0]);
    res = np.matmul(self.Homog, pt2);
    res /= res[2];
    return np.asarray([res[0], 0.0, res[1]]);
  
  # Very inaccurate, for some reason. Off by 30+ pixels.
  def ConvertWorldToImagePosition(self, pt):
    return cv2.projectPoints(pt, self.RotationVec, self.TranslationVec, self.CameraMatrix, self.DistCoeffs)[0][0][0];


from FindCourtCorners import FindCourtCorners
cap = cv2.VideoCapture('../UntrackedFiles/angle3_5.mp4')
_, frame = cap.read()
courtCorners = np.asarray([[1171,  471],
                           [1729,  525],
                           [ 930,  879],
                           [  51,  658]], dtype = "float");


kyleCam = Camera("kyle", courtCorners);
idx=0;

#print kyleCam.ConvertPixelToCourtPosition(np.asarray([1171,471]));







