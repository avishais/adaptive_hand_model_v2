#!/usr/bin/env python

import numpy as np
from gp import GaussianProcess
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
    K_manifold = 100
    df = DiffusionMap(sigma=1, embedding_dim=3, k=K)
else:
    K = 100
K_up = 100

print('Loading data...')
if discrete:
    if 1:
        Q = loadmat('../../../data/sim_data_discrete.mat')
        Qtrain = Q['D']
        is_start = Q['is_start'][0][0]; is_end = Q['is_end'][0][0]#-250
    else:
        Q = loadmat('../../../data/Ce_20_5.mat') # Real data from blue hand
        Qtrain = np.concatenate((Q['Xtest1'][0][0][0],Q['Xtraining']), axis=0)
        is_start = 0; is_end = Q['Xtest1'][0][0][0].shape[0]-1700
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
Qtrain = Qtrain[np.random.choice(Qtrain.shape[0], 30000, replace=False),:]
print('Loaded training data of ' + str(Qtrain.shape[0]) + '.')

state_action_dim = 6 
state_dim = 4

Xtrain = Qtrain[:,0:state_action_dim]
Ytrain = Qtrain[:,state_action_dim:]
Xtest = Qtest[:,0:state_action_dim]
Ytest = Qtest[:,state_action_dim:]

# Normalize
x_max_X = np.max(Xtrain, axis=0)
x_min_X = np.min(Xtrain, axis=0)
x_max_Y = np.max(Ytrain, axis=0)
x_min_Y = np.min(Ytrain, axis=0)

for i in range(state_dim):
    tmp = np.max([x_max_X[i], x_max_Y[i]])
    x_max_X[i] = tmp
    x_max_Y[i] = tmp
    tmp = np.min([x_min_X[i], x_min_Y[i]])
    x_min_X[i] = tmp
    x_min_Y[i] = tmp

for i in range(Xtrain.shape[1]):
    Xtrain[:,i] = (Xtrain[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
    Xtest[:,i] = (Xtest[:,i]-x_min_X[i])/(x_max_X[i]-x_min_X[i])
for i in range(Ytrain.shape[1]):
    Ytrain[:,i] = (Ytrain[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])
    Ytest[:,i] = (Ytest[:,i]-x_min_Y[i])/(x_max_Y[i]-x_min_Y[i])

Ytrain -= Xtrain[:,:state_dim]
Ytest -= Xtest[:,:state_dim]

print("Loading data to kd-tree...")
Xtrain_nn = Xtrain# * W
kdt = KDTree(Xtrain_nn, leaf_size=20, metric='euclidean')

def predict(sa):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    if useDiffusionMaps:
        X_nn, Y_nn = reduction(sa, X_nn, Y_nn)

    m = np.zeros(state_dim)
    s = np.zeros(state_dim)

    for i in range(state_dim):
        if i == 0:
            gp_est = GaussianProcess(X_nn[:,:4], Y_nn[:,i], optimize = True, theta=None)
            theta = gp_est.cov.theta
        else:
            gp_est = GaussianProcess(X_nn[:,:4], Y_nn[:,i], optimize = False, theta=theta)
        m[i], s[i] = gp_est.predict(sa[:4].reshape(1,-1)[0])

    return m, s

def batch_predict(SA):
    sa = np.mean(SA, 0)
    idx = kdt.query(sa.reshape(1,-1), k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    if useDiffusionMaps:
        X_nn, Y_nn = reduction(sa, X_nn, Y_nn)

    m = np.zeros((SA.shape[0], state_dim))
    s = np.zeros((SA.shape[0], state_dim))
    for i in range(state_dim):
        if i == 0:
            gp_est = GaussianProcess(X_nn[:,:4], Y_nn[:,i], optimize = True, theta=None)
            theta = gp_est.cov.theta
        else:
            gp_est = GaussianProcess(X_nn[:,:4], Y_nn[:,i], optimize = False, theta=theta)
        mm, ss = gp_est.batch_predict(SA[:,:4])
        m[:,i] = mm
        s[:,i] = np.diag(ss)

    return m, s

def reduction(sa, X, Y):
    inx = df.ReducedClosestSetIndices(sa, X, k_manifold=K_manifold)

    return X[inx,:][0], Y[inx,:][0]

# st = time.time()
# for i in range(100):
#     s = Xtest[127,:state_dim]
#     a = Xtest[127,state_dim:state_action_dim]
#     sa = np.concatenate((s,a)).reshape(-1,1)
#     ds_next, std_next = predict(sa)
# print("Computation time: " + str((time.time()-st)/100) + " sec.")
# exit(1)

st = time.time()
# GP propagation
print "Running GP."

s = Xtest[0,:state_dim]
Ypred_mean = s.reshape(1,state_dim)
Ypred_std = np.zeros((1,state_dim)).reshape(1,state_dim)

print("Running (open loop) path...")
for i in range(0, Xtest.shape[0]-0*210):
    print("Step " + str(i) + " of " + str(Xtest.shape[0]))
    a = Xtest[i,state_dim:state_action_dim]
    sa = np.concatenate((s,a)).reshape(-1,1)
    ds_next, std_next = predict(sa)
    s_next = s + ds_next
    # s_next = propagate(sa)
    print s_next
    # print std_next
    s = s_next
    Ypred_mean = np.append(Ypred_mean, s_next.reshape(1,state_dim), axis=0)
    Ypred_std = np.append(Ypred_std, std_next.reshape(1,state_dim), axis=0)

print("Computation time: " + str(time.time()-st) + " sec.")

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
plt.plot(Xtest[:,0], Xtest[:,1], 'k-')
plt.plot(Ypred_mean[:,0], Ypred_mean[:,1], 'r.-')
for i in range(Ypred_std.shape[0]):
    # print Ypred_std[i,:2]
    ell = Ellipse(xy=(Ypred_mean[i,0], Ypred_mean[i,1]), width=Ypred_std[i,0]*2, height=Ypred_std[i,1]*2, angle=0.)
    # ell.set_facecolor('none')
    ax.add_artist(ell)
# for i in range(Ypred.shape[0]-1):
#     plt.plot(np.array([Xtest[i,0], Ypred[i,0]]), np.array([Xtest[i,1], Ypred[i,1]]), 'r.-')
plt.axis('equal')
plt.title('GP_\Delta y')
plt.grid(True)
plt.show()