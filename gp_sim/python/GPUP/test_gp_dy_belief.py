#!/usr/bin/env python

import numpy as np
from gp import GaussianProcess
from gpup import UncertaintyPropagation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

# Particles prediction
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

# GPUP prediction
def predict(sa, sigma2):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    if useDiffusionMaps:
        X_nn, Y_nn = reduction(sa, X_nn, Y_nn)

    m = np.zeros(state_dim)
    s = np.zeros(state_dim)
    for i in range(state_dim):
        if i == 0:
            gpup_est = UncertaintyPropagation(X_nn[:,:state_dim], Y_nn[:,i], optimize = True, theta=None)
            theta = gpup_est.cov.theta
        else:
            gpup_est = UncertaintyPropagation(X_nn[:,:state_dim], Y_nn[:,i], optimize = False, theta=theta)
        m[i], s[i] = gpup_est.predict(sa[:state_dim].reshape(1,-1)[0], sigma2[:state_dim])

    # C = np.empty((state_dim,state_dim))
    # for i in range(state_dim):
    #     for j in range(state_dim):
    #         g = np.cov(X_nn[:,i],Y_nn[:,j])
    #         C[i,j] = g[1,0]

    return m, s, #np.diag(np.diag(s) + 2*C)

def reduction(sa, X, Y):
    inx = df.ReducedClosestSetIndices(sa, X, k_manifold=K_manifold)

    return X[inx,:][0], Y[inx,:][0]

st = time.time()

##########################################################################################################

s_start = Xtest[0,:state_dim]
sigma_start = np.array([0., 0., 0., 0.])+1e-8

######################################## GP propagation ##################################################
print "Running GP."

Np = 100 # Number of particles
s = s_start
S = np.tile(s, (Np,1)) + np.random.normal(0, sigma_start, (Np,state_dim))
Ypred_mean_gp = s.reshape(1,state_dim)
Ypred_std_gp = np.zeros((1,state_dim)).reshape(1,state_dim)

# ims = []
Pgp = []; 
print("Running (open loop) path...")
for i in range(0, Xtest.shape[0]):
    print("Step " + str(i) + " of " + str(Xtest.shape[0]))
    Pgp.append(S)
    a = Xtest[i,state_dim:state_action_dim]
    A = np.tile(a, (Np,1))
    SA = np.concatenate((S,A), axis=1)
    dS_next, std_next = batch_predict(SA)
    S_next = S + np.random.normal(dS_next, std_next)
    s_mean_next = np.mean(S_next, 0)
    s_std_next = np.std(S_next, 0)
    S = S_next

    Ypred_mean_gp = np.append(Ypred_mean_gp, s_mean_next.reshape(1,state_dim), axis=0)
    Ypred_std_gp = np.append(Ypred_std_gp, s_std_next.reshape(1,state_dim), axis=0)

######################################## GPUP propagation ###############################################

print "Running GPUP."

s = s_start
sigma2_x = sigma_start**2
Ypred_mean_gpup = s.reshape(1,state_dim)
Ypred_std_gpup = sigma2_x.reshape(1,state_dim)

print("Running (open loop) path...")
for i in range(0, Xtest.shape[0]):
    print("Step " + str(i) + " of " + str(Xtest.shape[0]))
    a = Xtest[i,state_dim:state_action_dim]
    sa = np.concatenate((s,a)).reshape(-1,1)
    ds_next, dsigma2_next = predict(sa, sigma2_x)
    s_next = s + ds_next
    sigma2_next = sigma2_x + dsigma2_next
    s = s_next
    sigma2_x = sigma2_next

    Ypred_mean_gpup = np.append(Ypred_mean_gpup, s_next.reshape(1,state_dim), axis=0)
    Ypred_std_gpup = np.append(Ypred_std_gpup, np.sqrt(sigma2_next).reshape(1,state_dim), axis=0)

######################################## Plot ###########################################################

print("Computation time: " + str(time.time()-st) + " sec.")

fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.plot(Xtest[:,0], Xtest[:,1], 'b-')

prtc_mean_line, = ax.plot([], [], '-r')
prtc, = ax.plot([], [], '.k')
prtc_mean, = ax.plot([], [], '*r')
patch_prtc = Ellipse(xy=(Ypred_mean_gp[0,0], Ypred_mean_gp[0,1]), width=Ypred_std_gp[0,0]*2, height=Ypred_std_gp[0,1]*2, angle=0., animated=True, edgecolor='y', linewidth=2., fill=False)
ax.add_patch(patch_prtc)

patch = Ellipse(xy=(Ypred_mean_gpup[0,0], Ypred_mean_gpup[0,1]), width=Ypred_std_gpup[0,0]*2, height=Ypred_std_gpup[0,1]*2, angle=0., animated=True, edgecolor='m', linewidth=2., linestyle='--', fill=False)
ax.add_patch(patch)
patch_mean, = ax.plot([], [], '--m')

def init():
    prtc.set_data([], [])
    prtc_mean.set_data([], [])
    prtc_mean_line.set_data([], [])
    patch_mean.set_data([], [])

    return prtc, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

def animate(i):

    S = Pgp[i]
    prtc.set_data(S[:,0], S[:,1])
    prtc_mean.set_data(Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    prtc_mean_line.set_data(Ypred_mean_gp[:i,0], Ypred_mean_gp[:i,1])
    patch_prtc.center = (Ypred_mean_gp[i,0], Ypred_mean_gp[i,1])
    patch_prtc.width = Ypred_std_gp[i,0]*2
    patch_prtc.height = Ypred_std_gp[i,1]*2

    patch.center = (Ypred_mean_gpup[i,0], Ypred_mean_gpup[i,1])
    patch.width = Ypred_std_gpup[i,0]*2
    patch.height = Ypred_std_gpup[i,1]*2
    patch_mean.set_data(Ypred_mean_gpup[:i,0], Ypred_mean_gpup[:i,1])

    return prtc, prtc_mean, prtc_mean_line, patch_prtc, patch, patch_mean,

ani = animation.FuncAnimation(fig, animate, frames=len(Pgp), init_func=init, interval=200, repeat_delay=3000, blit=True)
ani.save('belief.mp4', metadata={'artist':'Avishai Sintov','year':'2019'})

plt.show()