
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import time
# %matplotlib inline
# matplotlib.rcParams['figure.figsize'] = (12, 6)

N = 120
X = np.random.rand(N,1)
Y = np.sin(12*X) + 0.66*np.cos(25*X)# + np.random.randn(N,1)*0.1 + 3
plt.plot(X, Y, 'kx', mew=2)

k = gpflow.kernels.Matern52(1, lengthscales=0.3)
m = gpflow.models.GPR(X, Y, kern=k)
m.likelihood.variance = 0.01

# With linear mean function
# k = gpflow.kernels.Matern52(1, lengthscales=0.3)
# meanf = gpflow.mean_functions.Linear(1.0, 0.0)
# m = gpflow.models.GPR(X, Y, k, meanf)
# m.likelihood.variance = 0.01

def plot(m, i):
    xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)
    
    start = time.time()
    mean, var = m.predict_y(xx)
    end = time.time()
    print("Prediction time: %.1f seconds" % (end - start))

    plt.figure(i, figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'C0', lw=2)
    plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
    plt.xlim(-0.1, 1.1)
    

plot(m, 2)

# Maximum Likelihood estimation of \theta
# m.as_pandas_table() # See parameters before optimization
start = time.time()
gpflow.train.ScipyOptimizer().minimize(m)
end = time.time()
print("Opt. time: %.1f seconds" % (end - start))
plot(m, 3)
# m.as_pandas_table() # See parameters after optimization

plt.show()

