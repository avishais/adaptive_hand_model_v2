#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn

class KDE_failure():

    r = 0.05

    def __init__(self, discrete = True):

        # path = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/'
        path = '../data/'
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
        # file_pi = open('../data/transition_data_discrete_test.obj', 'wb')
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
        ni = 2
        T = np.where(self.done)[0]
        inx_fail = T[np.random.choice(T.shape[0], ni, replace=False)]
        T = np.where(np.logical_not(self.done))[0]
        inx_suc = T[np.random.choice(T.shape[0], ni, replace=False)]
        self.SA_test = np.concatenate((self.SA[inx_fail], self.SA[inx_suc]), axis=0)
        self.done_test = np.concatenate((self.done[inx_fail], self.done[inx_suc]), axis=0)
        
        self.SA = np.delete(self.SA, inx_fail, axis=0)
        self.SA = np.delete(self.SA, inx_suc, axis=0)
        self.done = np.delete(self.done, inx_fail, axis=0)
        self.done = np.delete(self.done, inx_suc, axis=0)

        

    def gaussian(self, s1, s2, b=1):
        x = np.linalg.norm(s1-s2)
        return np.exp(-x**2/(2*b**2))/(b*np.sqrt(2*np.pi))

    def KDE(self, sa, sa_nn):

        N = sa_nn.shape[0]
        if N==0:
            return 0.0

        K = 0
        for i in range(N):
            K += self.gaussian(sa_nn[i], sa)
        
        return K#/N

    def probability(self, sa):
        idx = self.kdt.query_radius(sa, r=self.r)[0]
        if len(idx)==0:
            return 1.0

        sa_nn = self.SA[idx]
        done_nn = self.done[idx]

        # sa_nn = np.array([[3,3],[1.5,2.3],[1.5,1.6]]).reshape(3,2)
        # done_nn = np.array([True,False,False])
        # sa = np.array([2.,2.]).reshape(1,2)

        # print sa_nn.shape, sa_nn
        # print sa

        K1 = self.KDE(sa, sa_nn[done_nn])
        K2 = self.KDE(sa, sa_nn)
        print K1, K2

        if K1/K2 > 1.:
            print '-------'
            print sa_nn
            print done_nn
            print sa
            c = 0
            for j in range(done_nn.shape[0]):
                if done_nn[j]:
                    c += 1
            print c, done_nn.shape[0]


            print '-------'

        return K1/K2


if __name__ == '__main__':
    
    K = KDE_failure()

    SA_test = K.SA_test

    for i in range(SA_test.shape[0]):
        print K.probability(SA_test[i].reshape(1,-1))
    





