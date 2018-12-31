import numpy as np
from gp import GaussianProcess
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import GPy

np.random.seed(2327864)

def func(x, v = 0.3):
    return x*np.sin(x)+np.random.normal(0, v)
    # return 3*x+4+np.random.normal(0, 0.01)

# x_data = np.linspace(0, 4, 5).reshape(-1,1)
x_data = np.random.uniform(0, 4, 7).reshape(-1,1)
y_data = np.array([func(i,0.2) for i in x_data]) #


x_real = np.linspace(0, 6, 100).reshape(-1,1)
y_real = np.array([func(i, 0) for i in x_real]) 

gp_est = GaussianProcess(x_data, y_data.reshape((-1,)), optimize = True, theta=None)

x_n = np.array([1.26])
m, s = gp_est.predict(x_n)

# print(m,s)

x_new = np.linspace(0, 6, 100).reshape(-1,1)
means = np.empty(100)
variances = np.empty(100)
for i in range(100):
    means[i], variances[i] = gp_est.predict(x_new[i])
# print(means)

# GPy
kernel = GPy.kern.RBF(input_dim=1, variance=0.9, lengthscale=0.5)
gpy = GPy.models.GPRegression(x_data, y_data, kernel)
gpy.optimize()
my = np.zeros(len(x_new))
sy = np.zeros(len(x_new))
for i in range(len(x_new)):
    my[i], sy[i] = gpy.predict(x_new[i].reshape(-1,1))
my_n, sy_n = gpy.predict(x_n.reshape(-1,1))
# plt.plot(x_new, my, '-c')


ax1=plt.subplot(1, 2, 1)
ax1.plot(x_data, y_data, '+k')
ax1.plot(x_real, y_real, '--k')
ax1.plot(x_n, m, '*y')
msl = (means.reshape(1,-1)[0]-variances)#.reshape(-1,1)
msu = (means.reshape(1,-1)[0]+variances)#.reshape(-1,1)[0]
ax1.plot(x_new, means,'-r')
ax1.fill_between(x_new.reshape(1,-1)[0], msl, msu)
plt.ylabel('f(x)')
plt.title('My GP')

ax2=plt.subplot(1, 2, 2)
ax2.plot(x_data, y_data, '+k')
ax2.plot(x_real, y_real, '--k')
ax2.plot(x_n, my_n, '*y')
msl = (my.reshape(1,-1)[0]-sy)#.reshape(-1,1)
msu = (my.reshape(1,-1)[0]+sy)#.reshape(-1,1)[0]
ax2.plot(x_new, my,'-r')
ax2.fill_between(x_new.reshape(1,-1)[0], msl, msu)
plt.ylabel('f(x)')
plt.title('GPy')

plt.show()