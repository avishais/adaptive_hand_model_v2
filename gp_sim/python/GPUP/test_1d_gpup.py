import numpy as np
import scipy.stats
from gp import GaussianProcess
from gpup import UncertaintyPropagation
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPy

np.random.seed(145453)

def func(x, v = 0.3):
    # return 0.3*x+np.random.normal(0, v)
    return x*np.sin(x)+np.random.normal(0, v)
    # return 3*x+4+np.random.normal(0, 0.01)

ax1 = plt.subplot2grid((3, 5), (0, 2), colspan=3, rowspan=2)

x_data = np.random.uniform(0, 4, 20).reshape(-1,1)
y_data = np.array([func(i, 0.2) for i in x_data]) #
plt.plot(x_data, y_data, '+k')

x_real = np.linspace(0, 6, 100).reshape(-1, 1)
y_real = np.array([func(i, 0) for i in x_real])
plt.plot(x_real, y_real, '--k')

gp_est = GaussianProcess(x_data, y_data.reshape((-1,)), optimize = True, theta = None)
gpup_est = UncertaintyPropagation(x_data, y_data.reshape((-1,)), optimize = True, theta = None, method = 3)

x_n = np.array([3.0]) # The mean of a normal distribution
var_n = np.diag([0.2**2]) # The covariance matrix (must be diagonal because of lazy programming)
m, s = gpup_est.predict(x_n, var_n)
# m, s = gp_est.predict(x_n)
print m, s
plt.errorbar(x_n, m, yerr=np.sqrt(s), ecolor = 'y')
plt.plot(x_n, m, '*y')

# exit(1)
x_new = np.linspace(0, 6, 100).reshape(-1,1)
means = np.empty(100)
variances = np.empty(100)
for i in range(100):
    means[i], variances[i] = gp_est.predict(x_new[i])
msl = (means.reshape(1,-1)[0]-np.sqrt(variances))#.reshape(-1,1)
msu = (means.reshape(1,-1)[0]+np.sqrt(variances))#.reshape(-1,1)[0]
ax1.plot(x_new, means,'-r')
ax1.fill_between(x_new.reshape(1,-1)[0], msl, msu)

# exit(1)

N = int(1e4)
X_belief = np.array([np.random.normal(x_n, np.sqrt(var_n)) for _ in range(N)]).reshape(N,1) #
ax4 = plt.subplot2grid((3, 5), (2, 2), colspan=3, rowspan=1)
plt.plot(X_belief, np.tile(0., N), '.k')
x = np.linspace(0, 6, 1000).reshape(-1,1)
plt.plot(x,scipy.stats.norm.pdf(x, x_n, np.sqrt(var_n)))
plt.xlabel('x')

ax2 = plt.subplot2grid((3, 5), (0, 0), colspan=1, rowspan=2)
means_b = np.empty(N)
variances_b = np.empty(N)
for i in range(N):
    means_b[i], variances_b[i] = gp_est.predict(X_belief[i])
Y_belief = np.array([np.random.normal(means_b[i], np.sqrt(variances_b[i])) for i in range(N)]).reshape(N,1) #
plt.plot(np.tile(0., N), Y_belief, '.k')
plt.ylabel('p(y)')

ylim = ax1.get_ylim()
mu_Y = np.mean(Y_belief)
sigma2_Y = np.std(Y_belief)
y = np.linspace(ylim[0], ylim[1], 1000).reshape(-1,1)
plt.plot(scipy.stats.norm.pdf(y, mu_Y, sigma2_Y), y, '-b')
plt.plot(scipy.stats.norm.pdf(y, m, np.sqrt(s)), y, ':r')

ax3 = plt.subplot2grid((3, 5), (0, 1), rowspan=2)
plt.hist(means_b, bins=20, orientation='horizontal')

ax2.set_ylim(ylim)
ax3.set_ylim(ylim)

plt.show()