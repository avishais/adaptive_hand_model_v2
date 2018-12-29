#!/usr/bin/env python

import sys
sys.path.insert(0, './gpuppy/')

import numpy as np
from GaussianProcess import GaussianProcess
from Covariance_original import GaussianCovariance
from UncertaintyPropagation import UncertaintyPropagationApprox, UncertaintyPropagationExact, UncertaintyPropagationMC
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pickle
from sklearn.neighbors import KDTree #pip install -U scikit-learn
from diffusionMaps import DiffusionMap
from scipy.io import loadmat
import time
from sklearn.preprocessing import StandardScaler

np.random.seed(10)

saved = False
discrete = True
useDiffusionMaps = False

# Number of NN
if useDiffusionMaps:
    K = 1000
    K_manifold = 200
    df = DiffusionMap(sigma=1, embedding_dim=2, k=K)
else:
    K = 200 
K_up = 100

print('Loading data...')
if discrete:
    Q = loadmat('../../data/sim_data_discrete.mat')
    Qtrain = Q['D']
    is_start = Q['is_start'][0][0]; is_end = Q['is_end'][0][0]#-250
else:
    Q = loadmat('../../data/sim_data_cont.mat')
    Qtrain = Q['D']
    is_start = Q['is_start'][0][0]; is_end = Q['is_end'][0][0]-100
# scaler = StandardScaler()
# Qtrain = scaler.fit_transform(Qtrain)
# x_mean = scaler.mean_
# x_std = scaler.scale_
Qtest = Qtrain[is_start:is_end,:]
Qtrain = np.concatenate((Qtrain[:is_start,:], Qtrain[is_end:,:]), axis=0)
Qtrain = Qtrain[np.random.choice(Qtrain.shape[0], 300000, replace=False),:]
print('Loaded training data of ' + str(Qtrain.shape[0]) + '.')

state_action_dim = 6 
state_dim = 4

Xtrain = Qtrain[:,0:state_action_dim]
Ytrain = Qtrain[:,state_action_dim:]
Xtest = Qtest[:,0:state_action_dim]
Ytest = Qtest[:,state_action_dim:]

# Normalize
# x_max_X = np.max(Xtrain, axis=0)
# x_min_X = np.min(Xtrain, axis=0)
# x_max_Y = np.max(Ytrain, axis=0)
# x_min_Y = np.min(Ytrain, axis=0)

# for i in range(state_dim):
#     tmp = np.max([x_max_X[i], x_max_Y[i]])
#     x_max_X[i] = tmp
#     x_max_Y[i] = tmp
#     tmp = np.min([x_min_X[i], x_min_Y[i]])
#     x_min_X[i] = tmp
#     x_min_Y[i] = tmp

# for i in range(Xtrain.shape[1]):
#     Xtrain[:,i] = (Xtrain[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
#     Xtest[:,i] = (Xtest[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
# for i in range(Ytrain.shape[1]):
#     Ytrain[:,i] = (Ytrain[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])
#     Ytest[:,i] = (Ytest[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])

print("Loading data to kd-tree...")
Xtrain_nn = Xtrain# * W
kdt = KDTree(Xtrain_nn, leaf_size=20, metric='euclidean')
# K = 200
# K_many = 500



def predict(sa, theta_min):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    if useDiffusionMaps:
        X_nn, Y_nn = reduction(sa, X_nn, Y_nn)

    m = np.zeros(state_dim)
    s = np.zeros(state_dim)
    for i in range(0,state_dim):
        gp_est = GaussianProcess(X_nn[:,:4], Y_nn[:,i], GaussianCovariance(), theta_min=None)
        m[i], s[i] = gp_est.estimate(sa[0][:4].reshape(1,-1))
    return m, s

def predict_many(SA, X, Y): # Assume that the test points are near each other (Gaussian distributed...)
    sa = np.mean(SA, 0).reshape(1,-1)
    idx = kdt.query(sa, k=K_many, return_distance=False)
    X_nn = X[idx,:].reshape(K_many, Dx)
    Y_nn = Y[idx,:].reshape(K_many, Dy)

    m = np.empty([SA.shape[0], Dy])
    s = np.empty([SA.shape[0], Dy])
    for i in range(Dy):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance())
        m[:,i], s[:,i] = gp_est.estimate_many(SA)
    return m, s

def UP(sa_mean, sa_Sigma):
    idx = kdt.query(sa.T, k=K_up, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K_up, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K_up, state_dim)

    m = np.empty(state_dim)
    s = np.empty(state_dim)
    for i in range(state_dim):
        gp_est = GaussianProcess(X_nn, Y_nn[:,i], GaussianCovariance(), theta_min=theta_min)
        up = UncertaintyPropagationExact(gp_est)
        m[i], s[i] = up.propagate_GA(sa_mean.reshape((-1,)), sa_Sigma)
    return m, s

def propagate(sa):
    mu, sigma = predict(sa)
    # s_next = np.random.normal(mu, sigma, state_dim)

    return mu#s_next

def reduction(sa, X, Y):
    inx = df.ReducedClosestSetIndices(sa, X, k_manifold=K_manifold)

    return X[inx,:][0], Y[inx,:][0]

def get_global_theta():
    Theta_min = []
    for i in range(0,state_dim):
        gp_est = GaussianProcess(Xtrain[:,:4], Ytrain[:,i], GaussianCovariance(), globalTheta=True)
        Theta_min.append(gp_est.theta_min)
        del gp_est

    return Theta_min

def RMS(X, Y):
    d = 0
    for i in range(X.shape[0]):
        d += np.linalg.norm(X[i,:4]-Y[i,:4])**2
    return np.sqrt(d/X.shape[0])


###

start = time.time()

for k in range(2):

    try:
        # GP propagation
        if 1:
            s = Xtest[0,:state_dim]
            Ypred_mean = s.reshape(1,state_dim)
            Ypred_std = np.zeros((1,state_dim)).reshape(1,state_dim)
            theta_min = np.random.random((1,6))*20-10
            print theta_min

            for i in range(4):#Xtest.shape[0]):
                a = Xtest[i,state_dim:state_action_dim]
                sa = np.concatenate((s,a)).reshape(-1,1)
                s_next, std_next = predict(sa, theta_min=theta_min.reshape((-1,)))
                # s_next = propagate(sa)
                print s_next
                s = s_next
                Ypred_mean = np.append(Ypred_mean, s_next.reshape(1,state_dim), axis=0)
                Ypred_std = np.append(Ypred_std, std_next.reshape(1,state_dim), axis=0)

        # GPUP propagation
        if 0:
            s = Xtest[0,:state_dim]
            m = np.array([0.**2, 0.**2, 0.**2, 0.**2])
            m_u = np.array([0.**2, 0.**2])
            Ypred_mean = s.reshape(1,state_dim)
            Ypred_std = np.sqrt(m).reshape(1,state_dim)

            for i in range(Xtest.shape[0]):
                a = Xtest[i,state_dim:state_action_dim]
                sa = np.concatenate((s,a)).reshape(-1,1)
                m = np.diag(np.concatenate((m, m_u), axis=0))
                s_next_mean, s_next_var = UP(sa, m)
                s = np.copy(s_next_mean)
                m = np.copy(s_next_var)
                Ypred_mean = np.append(Ypred_mean, s.reshape(1,state_dim), axis=0)
                Ypred_std = np.append(Ypred_std, np.sqrt(m).reshape(1,state_dim), axis=0)
    except:
        print("Error!!!")
        # exit(1)
        continue

    S = 0#RMS(Xtest[:,:4], Ypred_mean)
    print("Trial " + str(k) + " with error " + str(S))

    F = open('brute.txt','a') 
    for j in range(6):
        F.write("%f "%theta_min[0][j])
    F.write("%f\n"%S)
    F.close()

    fig = plt.figure(0)
    plt.plot(Xtest[:,0], Xtest[:,1], 'k-')
    plt.plot(Ypred_mean[:,0], Ypred_mean[:,1], 'r.-')
    plt.axis('equal')
    plt.title('GPUP')
    plt.grid(True)
    plt.show()

end = time.time()


print("Calc. time: " + str(end - start) + " sec.")

