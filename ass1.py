import numpy as np
import matplotlib.pyplot as plt

# Scenario 1


# (a) - MMSE Estimator

num = 2500

h = 5
mu_theta = 7
var_theta = 1
var_noise = 8
tau_noise = 1.0/var_noise
tau_theta = 1.0/var_theta

x = np.zeros(shape=(1, num))
theta = np.random.normal(loc=mu_theta, scale=np.sqrt(var_theta), size =(num, ))

x = h * theta + np.random.normal(loc=0.0, scale = np.sqrt(var_noise), size=(num,))
k = np.arange(1, len(x) + 1, step=1)
x_bar = np.cumsum(x)/k

mu_theta_est = (k*tau_noise*x_bar+tau_theta*mu_theta)/((k*tau_noise+tau_theta)*h)

mu_theta_est_temp = np.expand_dims(mu_theta_est, 0)

mse_error = np.average((np.repeat(np.asarray([theta]), num, axis = 0) - mu_theta_est_temp.T)**2, 1)

plt.figure()
plt.plot(mu_theta_est)
plt.title("MSE Estimate of $\mu_{\\theta}$")
plt.xlabel("Number of Iterations")
plt.figure()
plt.plot(mse_error)
plt.title("MSE Estimate Error")
plt.xlabel("Number of Iterations")
plt.show()



# (b) - ML Estimator

x = h * theta + np.random.normal(loc=0.0, scale = np.sqrt(var_noise), size=(num,))
k = np.arange(1, len(x) + 1, step=1)
x_bar = np.cumsum(x)/k
mu_theta_est = x_bar/h
mu_theta_est_temp = np.expand_dims(mu_theta_est, 0)
ml_error = np.average((np.repeat(np.asarray([theta]), num, axis = 0) - mu_theta_est_temp.T)**2, 1)

plt.figure()
plt.plot(mu_theta_est)
plt.title("ML Estimate of $\mu_{\\theta}$")
plt.xlabel("Number of Iterations")

plt.figure()
plt.plot(ml_error)
plt.xlabel("Number of Iterations")
plt.title("ML Estimate Error")
plt.show()



