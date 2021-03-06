import numpy as np
from skgpuppy.GaussianProcess import GaussianProcess
from skgpuppy.Covariance import GaussianCovariance
from skgpuppy.UncertaintyPropagation import UncertaintyPropagationApprox, UncertaintyPropagationExact
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPy

np.random.seed(10)

def func(x):
    return x*np.sin(x)+np.random.normal(0, 0.3)
    # return 3*x+4+np.random.normal(0, 0.01)

ax1 = plt.subplot2grid((3, 5), (0, 2), colspan=3, rowspan=2)

# x_data = np.linspace(0, 4, 5).reshape(-1,1)
x_data = np.random.uniform(0, 4, 7).reshape(-1,1)
y_data = np.array([func(i) for i in x_data]) #
plt.plot(x_data, y_data, '+k')
plt.ylabel('f(x)')

x_real = np.linspace(0, 4, 100).reshape(-1,1)
y_real = np.array([func(i) for i in x_real]) 
# plt.plot(x_real, y_real, '-g')

gp_est = GaussianProcess(x_data, y_data, GaussianCovariance())

x_n = np.array([1.26])
m, s = gp_est.estimate(x_n)
plt.plot(x_n, m, '*r')

print(m,s)



x_new = np.linspace(0, 6, 100).reshape(-1,1)
means, variances = gp_est.estimate_many(x_new)
# print(means)

# GPy
# kernel = GPy.kern.RBF(input_dim=1, variance=10.9, lengthscale=0.5)
# gpy = GPy.models.GPRegression(x_data, y_data, kernel)
# my = np.zeros(len(x_new))
# sy = np.zeros(len(x_new))
# for i in range(len(x_new)):
#     my[i], sy[i] = gpy.predict(x_new[i].reshape(-1,1))
# my_n, sy_n = gpy.predict(x_n.reshape(-1,1))
# plt.plot(x_n, my_n, '*c')
# plt.plot(x_new, my, '-c')

msl = (means.reshape(1,-1)[0]-variances)#.reshape(-1,1)
msu = (means.reshape(1,-1)[0]+variances)#.reshape(-1,1)[0]
plt.plot(x_new, means,'-r')
plt.fill_between(x_new.reshape(1,-1)[0], msl, msu)

plt.show()

exit(1)

# print msu

print "----------------------------------------------"


mean = np.array([3.0]) # The mean of a normal distribution
Sigma = np.diag([0.1**2]) # The covariance matrix (must be diagonal because of lazy programming)

up = UncertaintyPropagationExact(gp_est)

out_mean, out_variance = up.propagate_GA(mean, Sigma)
print(out_mean, out_variance)
# plt.errorbar(mean, out_mean, yerr=np.sqrt(out_variance))

print "----------------------------------------------"


N = int(1e4)
X_belief = np.array([np.random.normal(mean, np.sqrt(Sigma)) for _ in range(N)]).reshape(N,1) #
ax4 = plt.subplot2grid((3, 5), (2, 2), colspan=3, rowspan=1)
plt.plot(X_belief, np.tile(0., N), '.k')
x = np.linspace(0, 6, 1000).reshape(-1,1)
plt.plot(x,mlab.normpdf(x, mean, np.sqrt(Sigma)))
plt.xlabel('x')


ax2 = plt.subplot2grid((3, 5), (0, 0), colspan=1, rowspan=2)
means_b, variances_b = gp_est.estimate_many(X_belief)
Y_belief = np.array([np.random.normal(means_b[i], np.sqrt(variances_b[i])) for i in range(N)]).reshape(N,1) #
plt.plot(np.tile(0., N), Y_belief, '.k')
plt.ylabel('p(y)')

ylim = ax1.get_ylim()
mu_Y = np.mean(Y_belief)
sigma2_Y = np.std(Y_belief)
y = np.linspace(ylim[0], ylim[1], 1000).reshape(-1,1)
plt.plot(mlab.normpdf(y, mu_Y, sigma2_Y), y, '-b')
plt.plot(mlab.normpdf(y, out_mean, np.sqrt(out_variance)), y, ':r')

ax3 = plt.subplot2grid((3, 5), (0, 1), rowspan=2)
plt.hist(means_b, bins=20, orientation='horizontal')

ax2.set_ylim(ylim)
ax3.set_ylim(ylim)

plt.show()