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


saved = False
discrete = True
useDiffusionMaps = False

# Number of NN
if useDiffusionMaps:
    K = 1000
    K_manifold = 200
    df = DiffusionMap(sigma=1, embedding_dim=2, k=K)
else:
    K = 100 
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

# theta_min = np.array([-1.49409838, -2.88039275, -0.44337252,  0.16309625,  0.11204813, -0.55605511])*np.random.random((6,))*4
theta_min = np.array([-2.14, -3.5, -0.44337252,  0.16309625,  0.11204813, -0.55605511, 0.15, 0.4])*np.random.random((8,))*4.


def predict(sa):
    idx = kdt.query(sa.T, k=K, return_distance=False)
    X_nn = Xtrain[idx,:].reshape(K, state_action_dim)
    Y_nn = Ytrain[idx,:].reshape(K, state_dim)

    if useDiffusionMaps:
        X_nn, Y_nn = reduction(sa, X_nn, Y_nn)

    m = np.zeros(state_dim)
    s = np.zeros(state_dim)
    for i in range(0,state_dim):
        gp_est = GaussianProcess(X_nn[:,:4], Y_nn[:,i], GaussianCovariance(), theta_min=theta_min)
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

###

# Theta_min = get_global_theta()

start = time.time()

# GP propagation
if 0:
    if (saved):
        print('Loading saved path...')
        # Getting back the objects:
        with open('saved_GPUP.pkl') as f:  
            Xtest, Ypred = pickle.load(f)   
    else:
        s = Xtest[0,:state_dim]
        Ypred_mean = s.reshape(1,state_dim)
        Ypred_std = np.zeros((1,state_dim)).reshape(1,state_dim)

        print("Running (open loop) path...")
        for i in range(Xtest.shape[0]):
            print("Step " + str(i) + " of " + str(Xtest.shape[0]))
            a = Xtest[i,state_dim:state_action_dim]
            sa = np.concatenate((s,a)).reshape(-1,1)
            s_next, std_next = predict(sa)
            # s_next = propagate(sa)
            print s_next
            print std_next
            s = s_next
            Ypred_mean = np.append(Ypred_mean, s_next.reshape(1,state_dim), axis=0)
            Ypred_std = np.append(Ypred_std, std_next.reshape(1,state_dim), axis=0)

        # with open('saved_GPUP.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([Xtest, Ypred], f)

# GPUP propagation
if 1:
    s = Xtest[0,:state_dim]
    m = np.array([0.**2, 0.**2, 0.**2, 0.**2])
    m_u = np.array([0.**2, 0.**2])
    Ypred_mean = s.reshape(1,state_dim)
    Ypred_std = np.sqrt(m).reshape(1,state_dim)

    print("Running (open loop) path...")
    for i in range(Xtest.shape[0]):
        print("Step " + str(i) + " of " + str(Xtest.shape[0]))
        a = Xtest[i,state_dim:state_action_dim]
        sa = np.concatenate((s,a)).reshape(-1,1)
        m = np.diag(np.concatenate((m, m_u), axis=0))
        s_next_mean, s_next_var = UP(sa, m)
        print(s_next_mean, s_next_var)
        s = np.copy(s_next_mean)
        m = np.copy(s_next_var)
        Ypred_mean = np.append(Ypred_mean, s.reshape(1,state_dim), axis=0)
        Ypred_std = np.append(Ypred_std, np.sqrt(m).reshape(1,state_dim), axis=0)

end = time.time()

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
plt.title('GPUP')
plt.grid(True)
plt.show()

print("Calc. time: " + str(end - start) + " sec.")

