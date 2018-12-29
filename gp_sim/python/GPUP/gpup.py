
'''
Implementation based on:
Agathe Girard PhD thesis: Approximate Methods for Propagation of Uncertainty with Gaussian Process Models
                          http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.1826&rep=rep1&type=pdf

Marc Peter Deisenroth PhD thesis: Efficient Reinforcement Learning using Gaussian Processes
                          https://www.doc.ic.ac.uk/~mpd37/publications/thesis_mpd_corrected.pdf

'''


import numpy as np
from cov import Covariance


class UncertaintyPropagation(object):
    # Implements Exact moments, Girard, Pg. 43

    def __init__(self, X, Y, optimize = True, theta = None):
        
        self.X = X
        self.Y_mean = np.mean(Y)
        self.Y = Y - self.Y_mean

        self.cov = Covariance(self.X, self.Y, theta = theta, optimize = optimize)

        self.method = 2 # 1 - girard, 2 - Deisenroth

    def predict(self, mu_x, sigma_x):

        d = self.cov.d

        self.beta = np.dot(self.cov.Kinv, self.Y)
        self.W = np.diag(self.cov._get_w())
        self.Winv = np.linalg.inv(self.W)
        self.Winv2 = np.linalg.inv(self.W/2)
        self.Sigma_x = np.diag(sigma_x)

        if self.method == 1:
            # Girard Eq. 3.40, 3.41
            self.DeltaInv = self.Winv - np.linalg.inv(self.W + self.Sigma_x) 
            self.M = 1 / np.sqrt( np.linalg.det( np.eye(d) + self.Winv * self.Sigma_x ) )
            self.DeltaInv2 = 2*self.Winv - np.linalg.inv(0.5*self.W + self.Sigma_x) 
            self.M2 = 1 / np.sqrt( np.linalg.det( self.Winv2 * self.Sigma_x + np.eye(d) )  )
        else:
            # Deisenroth Eq. 2.34
            self.DeltaInv = np.linalg.inv(self.W + self.Sigma_x) 
            self.M = 1 / np.sqrt( np.linalg.det( self.Sigma_x * self.Winv + np.eye(d) ) )
            self.DeltaInv2 = np.linalg.inv(self.Sigma_x + 0.5*self.W) * self.Sigma_x * self.Winv
            self.M2 = 1 / np.sqrt( np.linalg.det( 2*self.Sigma_x*self.Winv + np.eye(d) ) )

        mean = self.predict_mean(mu_x)
        variance = self.predict_variance(mu_x, mean)

        return mean, variance

    def _C_corr(self, x, xi):

        diff = x - xi

        if self.method == 1:
            # Girard Eq. 3.40
            return self.M * np.exp( 0.5 * np.dot( diff.T, np.dot( self.DeltaInv, diff ) ) ) 
        else:
            # Deisenroth Eq. 2.34
            v = self.cov._get_v()
            return v * self.M * np.exp( - 0.5 * np.dot( diff, np.dot( self.DeltaInv, diff ) ) ) 

    def _C_corr_2(self, x, x_):

        diff = x - x_

        if self.method == 1:
            # Girard Eq. 3.41
            return self.M2 * np.exp( 0.5 * np.dot( diff.T, np.dot( self.DeltaInv2, diff ) ) ) 
        else:
            # Deisenroth Eq. 2.42
            return self.M2 * np.exp( np.dot( diff.T, np.dot( self.DeltaInv2, diff ) ) ) 
        
    def predict_mean(self, mu_x):

        if self.method == 1:
            # Girard Eq. 3.40
            mean = 0.
            for i in range(self.cov.N):
                mean += self.beta[i] * self.cov.Gcov(mu_x, self.X[i,:]) * self._C_corr(mu_x, self.X[i,:]) 
        else:
            # Deisenroth Eq. 2.34
            q = np.empty((self.cov.N,1))
            for i in range(self.cov.N):
                q[i] = self._C_corr(mu_x, self.X[i,:])
            mean = np.dot(self.beta, q)

        return mean + self.Y_mean

    def predict_variance(self, mu_x, mean):

        if self.method == 1:
            # Girard Eq. 3.43
            S = 0.
            for i in range(self.cov.N):
                for j in range(self.cov.N):
                    x_ = (self.X[i,:] + self.X[j,:]) / 2.

                    Ci = self.cov.Gcov(mu_x, self.X[i,:])
                    Cj = self.cov.Gcov(mu_x, self.X[j,:])
                    C_corr_2 = self._C_corr_2(mu_x, x_)

                    S += (self.cov.Kinv[i,j] - self.beta[i]*self.beta[j]) * Ci * Cj * C_corr_2

            var = self.cov.Gcov(mu_x, mu_x) - S - mean**2 # Girard 3.43

        else:
            # Deisenroth Eq. 2.41
            Q = np.empty((self.cov.N, self.cov.N))
            for i in range(self.cov.N):
                for j in range(self.cov.N):
                    x_ = (self.X[i,:] + self.X[j,:]) / 2.

                    Ci = self.cov.Gcov(mu_x, self.X[i,:])
                    Cj = self.cov.Gcov(mu_x, self.X[j,:])
                    C_corr_2 = self._C_corr_2(mu_x, x_)

                    Q[i,j] = Ci * Cj * C_corr_2

            var = self.cov.Gcov(mu_x, mu_x) - np.trace( self.cov.Kinv * Q ) + np.dot( self.beta, np.dot(Q, self.beta)) - mean**2


        return var



