#!/usr/bin/env python

import numpy as np
# from numpy.linalg import cholesky
from scipy.optimize import minimize



class Covariance(object):

    def __init__(self, X, Y, theta=None, optimize = True):
        self.Set_Data(X, Y)
        
        if theta is None:
            self.Set_Initial_Theta()
        else:
            self.theta = theta

        # optimize hyper-parameters
        if optimize:
            self.optimize()
        
        self.K = self.cov_matrix_ij(self.X, self.X)
        # self.K = self.cov_matrix()
        self.Kinv = np.linalg.pinv(self.K)   

    def Set_Data(self, X, Y):
        self.X = X
        self.Y = Y
        self.d = self.X.shape[1]
        self.N = self.X.shape[0]

    def Set_Initial_Theta(self):

		theta = np.ones(2 + self.d)
		theta[0] = np.log(np.var(self.Y))  #size
		theta[1] = np.log(np.var(self.Y)/4)  #noise
		theta[2:] = -2*np.log((np.max(self.X,0)-np.min(self.X,0)+1e-3)/2.0) # w 
		self.theta = theta

    def _get_v(self):
        return np.exp(self.theta[0])
    def _get_vt(self):
        return np.exp(self.theta[1])
    def _get_w(self):
        return np.exp(self.theta[2:])
    def _get_theta(self):
        return np.exp(self.theta[0]), np.exp(self.theta[1]), np.exp(self.theta[2:])


    def Gcov(self,xi,xj):
        # Computes a scalar covariance of two samples

		v, vt, w = self._get_theta()

		diff = xi - xj

        #slighly dirty hack to determine whether i==j
		return v * np.exp(-0.5 * (np.dot(diff.T, w* diff))) + (vt if (xi == xj).all() else 0)

    def cov_matrix(self):
        vt = self._get_vt()

        # Computes each component indivisually
        K = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                K[i,j] = self.Gcov(self.X[i,:], self.X[j,:])

        return K + vt*np.eye(K.shape[0])

    def cov_matrix_ij(self, Xi, Xj): 
        # This is more efficient as it computes by matrix multiplication

        v, vt, w = self._get_theta()
        
        x1 = np.copy(Xi)
        x2 = np.copy(Xj)
        n1,_ = np.shape(x1)
        n2 = np.shape(x2)[0]
        x1 = x1 * np.tile(np.sqrt(w),(n1,1))
        x2 = x2 * np.tile(np.sqrt(w),(n2,1))

        K = -2*np.dot(x1,x2.T)

        K += np.tile(np.atleast_2d(np.sum(x2*x2,1)),(n1,1))
        K += np.tile(np.atleast_2d(np.sum(x1*x1,1)).T,(1,n2))
        K = v*np.exp(-0.5*K) + vt*np.eye(K.shape[0])

        return K

    def optimize(self):
        bounds = None#[(-20.,20.) for _ in range(self.d+2)]
        res = minimize(self.neg_log_marginal_likelihood, self.theta, method='l-bfgs-b', bounds=bounds,tol=1e-12, options={'disp':False,'eps':0.001})
        self.theta = res['x']

        # from Utilities import minimize
        # self.theta = minimize(self.neg_log_marginal_likelihood, self.theta, bounds = bounds,constr = None,fprime = None, method=["l_bfgs_b"])#all, tnc, l_bfgs_b, cobyla, slsqp, bfgs, powell, cg, simplex or list of some of them

        # print "Optimized hyper-parameters with cost function " + str(res['fun']) + "."
        print "Theta is now " + str(self.theta)

    def neg_log_marginal_likelihood(self,l):
        K = self.cov_matrix_ij(self.X, self.X)
        Kinv = np.linalg.pinv(K)

        return 0.5*np.dot(self.Y, np.dot(Kinv, self.Y)) + 0.5*np.log(np.linalg.det(K)) + 0.5*self.N*np.log(2*np.pi)

