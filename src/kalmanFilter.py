# kalman filter class to take care of filtering math

# ball state: position and velocity in 3d
# ball dynamics: x,z velocity should be constant, y velocity dependent on acceleration
# x_k1 = x_k + deltaT*v_x   ...after hitting the ball this is true
# z_k1 = z_k + deltaT*v_z   ...after hitting the ball this is true
# y_k1 = y_k + deltaT*v_y + 1/2*gravity*deltaT^2
# if y_k1 < 0 --> y_k1 = abs(y_k1) ...estimate perfect rebound/impulse
        # v_yk1 = -v_yk ...reverse velocity, should be positive again
# v_xk1 = v_xk
# z_xk1 = z_xk

# can be represented as a linear system where mu_k1 = A_k*mu_k + B_k*u_k
# A_k = [1 0 0 dt 0  0
#        0 1 0 0  dt 0
#        0 0 1 0  0  dt
#        0 0 0 1  0  0
#        0 0 0 0  1  0
#        0 0 0 0  0  1]
# B_k*u_k = [0
#            0.5*self.gravity*self.dt*self.dt
#            0
#            0
#            self.gravity*dt
#            0]

# measurement model: y_k = only positions = C_k*x_k
# C_k = [1 0 0 0 0 0
#        0 1 0 0 0 0
#        0 0 1 0 0 0]

import numpy as np

class KalmanFilter(object):
    def __init__(self, framerate):
        # state estimation parameters for ball
        # treat as a point mass for simplicity
        self.gravity = -9.8   # m/s^2
        self.ballMass = 0.058 # kg
        self.dt = 1.0/framerate # framerate should be 60fps, d_t = 1/60 sec per frame

        # maybe try particle filter with unknown correspondence?
        # now need to come up with initial ball estimates
        self.mu_k = np.zeros((6,1))
        self.sigma_k = np.eye(6) * 100.0
        self.Q = np.identity(6)*self.dt   # model noise covariance
        self.R = 0.05*np.identity(3)   # measurement noise covariance. keep to be threshold of ray intersect distance
        self.A_k = np.identity(6) + np.eye(6,k=3)*self.dt
        self.Bk_uk = np.array([[0], [0.5*self.gravity*self.dt*self.dt], [0],[0],[self.gravity*self.dt], [0]])
        self.C_k = np.hstack((np.identity(3),np.zeros((3,3))))

    # this actually sets self.mu_k and self.sigma_k, need to call update after
    def predict(self):
        # kalman filter prediction step, propogate dynamics
        # mu_k1 = f(mu_k) = A*mu_k + B_k*u_k
        mu_k1 = np.matmul(self.A_k,self.mu_k) + self.Bk_uk
        # now adjust for if bounce off ground:
        if mu_k1[1] < 0:
            # estimate rebound off ground, exactly reflects pos, vel/2
            mu_k1[1] = -mu_k1[1]
            mu_k1[4] = -0.5*mu_k1[4]
        sigma_k1 = np.matmul(np.matmul(self.A_k,self.sigma_k),self.A_k.T) + self.Q
        # return mu_k1, sigma_k1
        self.mu_k = mu_k1
        self.sigma_k = sigma_k1

    # y_k is the next measurment, should be ball position, x,y,z
    def update(self, y_k):
        # kalman filter update step, take into account measurement
        cscPr = np.matmul(np.matmul(self.C_k,self.sigma_k),self.C_k.T) + self.R
        cscPrInv = np.linalg.inv(cscPr)
        measErr = np.reshape(y_k,(3,1)) - np.matmul(self.C_k,self.mu_k)
        sigC = np.matmul(self.sigma_k, self.C_k.T)
        cSig = np.matmul(self.C_k, self.sigma_k)
        mu_k1 = self.mu_k + np.matmul(np.matmul(sigC,cscPrInv),measErr)
        sigma_k1 = self.sigma_k - np.matmul(np.matmul(sigC,cscPrInv),cSig)
        return mu_k1, sigma_k1

    def processMeas(self, measList,measCertainty):
        # gets a list of measurements and has to figure out most likely candidate?
        # things to consider, measurement Certainty is how far rays are apart
        # decreases covaraince uncertainty a lot
        # matches prediction value closest
        numMeas = len(measList)
        self.predict()
        if numMeas == 0:
            # no valid measurements, estimate ball location only by prediction:
            print 'no measurements received'
        if numMeas > 0:
            minDistToPred = float('inf')
            closestToPred_mu = []
            closesttoPred_sig = []
            # results in smallest covariance matrix?
            minSigNorm = np.linalg.norm(self.sigma_k,'fro')
            minSig_mu = []
            minSig_sig = []
            # smallest measurement distance?
            minMeasDist = float('inf')
            minMeas_mu = []
            minMeas_sig = []
            for i in range(0,numMeas):
                y_k = measList[i]
                mu_k1, sig_k1 = self.update(y_k)
                distToPred = np.linalg.norm(self.mu_k-mu_k1)
                if distToPred < minDistToPred:
                    closestToPred_mu = mu_k1
                    closestToPred_sig = sig_k1
                    minDistToPred = distToPred
                sigNorm = np.linalg.norm(sig_k1,'fro')
                if sigNorm < minSigNorm:
                    minSigNorm = sigNorm
                    minSig_mu = mu_k1
                    minSig_sig = sig_k1
                if measCertainty[i] < minMeasDist:
                    minMeasDist = measCertainty[i]
                    minMeas_mu = mu_k1
                    minMeas_sig = sig_k1
                    # minMeas_mu = y_k
            # if covariance fairly small, then already fairly certain in current ball position estimate
            # threshold to be tuned....
            if np.linalg.norm(self.sigma_k, 'fro') < 12:
                distFromPrev = np.linalg.norm(closestToPred_mu - self.mu_k)
                if distFromPrev < 7:
                    self.mu_k = closestToPred_mu
                    self.sigma_k = closestToPred_sig
                else:
                    print 'using prediction only', distFromPrev
            elif np.linalg.norm(self.sigma_k, 'fro') > 200:
                # really have no idea where are, reset to measurement?
                # self.mu_k = np.vstack((np.reshape(minMeas_mu, (3,1)),np.zeros((3,1))))
                self.mu_k = minMeas_mu
                # reset sigma_k
                #self.sigma_k = np.eye(6) * 100.0
                self.sigma_k = minMeas_sig
            else:
                self.mu_k = minSig_mu
                self.sig_k1 = minSig_sig
