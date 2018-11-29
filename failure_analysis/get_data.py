#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn

class KDE_failure():

    r = 0.05

    def __init__(self, discrete = True):

        # path = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/'
        path = './'
        mode = 'discrete' if discrete else 'cont'
        self.file_name = path + 'transition_data_' + mode + '_test.obj'

        self.load_data()

        self.inx_drop = np.where(self.done)
        n_drop = np.array(self.inx_drop).shape[1]

        # Set kD-tree
        self.kdt = KDTree(self.SA, leaf_size=100, metric='euclidean')
        print 'All set!'

    def load_data(self):
        print('Loading data from ' + self.file_name)
        with open(self.file_name, 'rb') as filehandler:
            self.memory = pickle.load(filehandler)
        print('Loaded transition data of size %d.'%len(self.memory))

        # M = self.memory[:100000]
        # print('Saving data...')
        # file_pi = open('transition_data_discrete_test.obj', 'wb')
        # pickle.dump(M, file_pi)
        # print('Saved transition data of size %d.'%len(M))
        # file_pi.close()

        self.states = np.array([item[0] for item in self.memory])
        self.actions = np.array([item[1] for item in self.memory])
        self.done = np.array([item[3] for item in self.memory])

        # Process data
        self.SA = np.concatenate((self.states, self.actions), axis=1)
        self.x_max = np.max(self.SA, axis=0)
        self.x_min = np.min(self.SA, axis=0)

        for i in range(self.SA.shape[1]):
            self.SA[:,i] = (self.SA[:,i]-self.x_min[i])/(self.x_max[i]-self.x_min[i])

        # Test data
        ni = 10
        T = np.where(self.done)[0]
        inx_fail = T[np.random.choice(T.shape[0], ni, replace=False)]
        T = np.where(np.logical_not(self.done))[0]
        inx_suc = T[np.random.choice(T.shape[0], ni, replace=False)]
        self.SA_test = np.concatenate((self.SA[inx_fail], self.SA[inx_suc]), axis=0)
        self.done_test = np.concatenate((self.done[inx_fail], self.done[inx_suc]), axis=0)
        
        print self.SA.shape[0]
        self.SA = np.delete(self.SA, inx_fail, axis=0)
        self.SA = np.delete(self.SA, inx_suc, axis=0)
        self.done = np.delete(self.done, inx_fail, axis=0)
        self.done = np.delete(self.done, inx_suc, axis=0)
        print self.SA.shape[0]

        print self.SA_test
        print self.done_test


    def gaussian(self, s1, s2, b=1):
        x = np.linalg.norm(sa-s2)
        return np.exp(-x**2/(2*b**2))/(b*np.sqrt(2*np.pi))

    def KDE(self, sa, sa_nn):

        N = sa_nn.shape[0]
        K = 0
        for _ in range(N):
            K += self.gaussian(sa_nn, sa)
        
        return K/N

    def probability(self, sa):
        idx = self.kdt.query_radius(sa, r=r)

        sa_nn = self.SA[idx]
        done_nn = self.done[idx]

        return self.KDE(sa, sa_nn[done_nn]) / self.KDE(sa, sa_nn)


if __name__ == '__main__':
    
    K = KDE_failure()
    





