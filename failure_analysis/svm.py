#!/usr/bin/env python

import numpy as np
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn
from sklearn import svm
from sklearn.preprocessing import StandardScaler

take_test_data = True
gen_new_data = False

class KDE_failure():

    r = 0.1

    def __init__(self, discrete = True):

        self.mode = 'discrete' if discrete else 'cont'

        self.load_data(gen_new_data)

        print 'All set!'

    def load_data(self, gen=True):

        path = '/home/pracsys/catkin_ws/src/rutgers_collab/src/sim_transition_model/data/'

        if gen:
            file_name = path + 'transition_data_' + self.mode + '.obj'
            print('Loading data from ' + file_name)
            with open(file_name, 'rb') as filehandler:
                self.memory = pickle.load(filehandler)
            print('Loaded transition data of size %d.'%len(self.memory))

            self.states = np.array([item[0] for item in self.memory])
            self.actions = np.array([item[1] for item in self.memory])
            self.done = np.array([item[3] for item in self.memory])

            # Process data
            self.SA = np.concatenate((self.states, self.actions), axis=1)

            # Sparser
            T = np.where(self.done)[0]
            inx_fail = T
            T = np.where(np.logical_not(self.done))[0]
            inx_suc = T[np.random.choice(T.shape[0], 10000, replace=False)]
            self.SA = np.concatenate((self.SA[inx_fail], self.SA[inx_suc]), axis=0)
            self.done = np.concatenate((self.done[inx_fail], self.done[inx_suc]), axis=0)

            with open(path + 'svm_data_' + self.mode + '.obj', 'wb') as f: 
                pickle.dump([self.SA, self.done], f)
            print('Saved svm data.')
        else:
            with open(path + 'svm_data_' + self.mode + '.obj', 'rb') as f: 
                self.SA, self.done = pickle.load(f)
            print('Loaded svm data.')
            

        # Normalize
        scaler = StandardScaler()
        self.SA = scaler.fit_transform(self.SA)
        self.x_mean = scaler.mean_
        self.x_std = scaler.scale_

        # Test data
        if take_test_data:
            ni = 40
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

        print 'Fitting SVM...'
        self.clf = svm.SVC( probability=True, class_weight='balanced', C=1.0 )
        self.clf.fit( list(self.SA), 1*self.done )
        print 'SVM ready with %d classes: '%len(self.clf.classes_) + str(self.clf.classes_)

    def probability(self, sa):

        # Normalize
        if not take_test_data:
            sa = (sa - self.x_mean) / self.x_std

        p = self.clf.predict_proba(sa)[0]
        return p, self.clf.predict(sa)


if __name__ == '__main__':
    
    K = KDE_failure(False)

    if take_test_data:
        SA_test = K.SA_test
        s = 0
        for i in range(SA_test.shape[0]):
            p, fail = K.probability(SA_test[i].reshape(1,-1))
            print p, K.done_test[i], fail
            s += 1 if fail == K.done_test[i] else 0
        print 'Success rate: ' + str(float(s)/SA_test.shape[0])
    





