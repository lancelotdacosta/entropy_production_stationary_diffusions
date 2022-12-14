'''
Illustration Helmholtz decomposition 2 way OU process
'''

import OU_Process.Functions.OU_process_functions as OU
import numpy as np
from numpy.linalg import inv, det
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.rcParams["text.usetex"] =True
plt.style.use('seaborn-white')
from scipy.stats import multivariate_normal
import seaborn as sns


np.random.seed(2)
cmap_Gaussian = 'Greys'

'''
Setting up the steady-state
'''

dim = 2  # dimension of state-space


std = 1  # define standard deviations of distributions we are sampling random numbers

# Define precision
Pi = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]) #random precision matrix
Pi = (Pi + Pi.T)/2
#Pi = np.array([2, 1, 1, 2]).reshape([dim, dim]) #selected precision matrix
# make sure Pi is positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)


'''
Setting up the OU process
'''

# volatility
sigma = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary volatility matrix
#sigma = np.zeros([dim,dim]) #no noise
# sigma = np.diag(np.random.normal(scale=std, size=dim)) # arbitrary diagonal volatility matrix
#sigma = np.array([2, 1.5, 0.5, 0]).reshape([dim, dim]) #selected non-degenerate noise
#sigma = np.array([1,0,1,0]).reshape([dim,dim]) #selected degenerate noise

# see whether noise is degenerate or not
print(f'det sigma = {det(sigma)}')

#diffusion tensor
D = sigma @ sigma.T / 2  # diffusion tensor

# solenoidal flow
Q = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary solenoidal flow
#Q = np.zeros([dim, dim])  # no solenoidal flow
#Q = np.array([0, 1.5, -1.5, 0]).reshape([dim, dim]) #selected solenoidal flow
Q = 5*(Q - Q.T)

# Drift matrix
B = (D + Q) @ Pi  # drift matrix
if np.any(spec(B) <= -10 ** (-5)):
    print(spec(B))
    raise TypeError("Drift should have non-negative spectrum")

# 1) We check it solves the Sylvester equation: BS + SB.T = 2D
# 2) we check that there are no numerical errors due to ill conditioning
error_sylvester = np.sum(np.abs(B @ S + S @ B.T - 2 * D))
error_inversion = np.sum(np.abs(S @ Pi - np.eye(dim)))
if np.round(error_sylvester, 7) != 0 or np.round(error_inversion, 7) != 0:
    raise TypeError("Sylvester equation not solved")
if np.sum(np.abs(inv(S) - Pi)) > 10 ** (-5):
    raise TypeError("Precision and inverse covariance are different")

# We check that the stationary covariance is indeed positive definite
if np.any(spec(S) <= 0):
    print(spec(S))
    raise TypeError("Stationary covariance not positive definite")

# Setting up the OU process
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create process


'''
OU process stationary simulation (full dynamic)
'''
print("full dynamic simulation")

epsilon = 10**(-4)  # time-step
T = 10 ** 4 #10 ** 4  # number of time-steps
#T = 5*10**4
N = 5  # number of trajectories


# Setting up the initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=[N]).reshape([dim, N])  # stationary initial condition

# sample paths
np.random.seed(2)
x = process.exact_simulation(x0, epsilon, T, N)  # run simulation

#update finish time if process finished early
T = x.shape[1]
print(f'T= {T}')

'''
Plotting trajectory full dynamic
'''
print("plotting full dynamic")


n = 0  # which sample path to show (between 0 and N-1)

#lim_x = 2.4
lim_y = 1.3

x_tick = np.linspace(-2, 2.5, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points
#x_tick = np.linspace(np.min(x[0, :, n]) - 0.5, np.max(x[0, :, n]) + 0.5, 105)  # x axis points
#y_tick = np.linspace(np.min(x[1, :, n]) - 0.5, np.max(x[1, :, n]) + 0.5, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

rv = multivariate_normal(cov= S) #random normal

plt.figure(2)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)  # plotting the free energy landscape

plt.suptitle('Full dynamic',fontsize=16)
plt.title(r'$dx_t = b_{rev}(x_t)dt+b_{irr}(x_t)dt+ \sigma(x_t)dw_t$',fontsize=14)

OU.plot_cool_colourline3(x[0, :, n].reshape(T), x[1, :, n].reshape(T), lw=0.5)
plt.savefig("Helmholtz_complete.png", dpi=100)


plt.figure(3)
plt.clf()
plt.plot(range(T), x[0, :, n].reshape(T))
plt.plot(range(T), x[1, :, n].reshape(T))


'''
Conservative (time irreversible) simulation
'''
print("conservative dynamic simulation")


B = Q @ Pi  # drift matrix

# Setting up the OU process
conservative_process = OU.OU_process(dim=dim, friction=B, volatility=np.zeros([dim, dim]))  # create process

np.random.seed(2)
x = conservative_process.exact_simulation(x0, epsilon, int(T/2), N)  # run simulation

#update finish time if process finished early
print(f'T= {x.shape[1]}')


'''
Plot trajectory conservative simulation
'''
print("plotting conservative dynamic")


plt.figure(4)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)
plt.suptitle('Time-irreversible part',fontsize=16)
plt.title(r'$dx_t = b_{irr}(x_t)dt$',fontsize=14)
OU.plot_cool_colourline3(x[0, :, n].reshape(int(T/2)), x[1, :, n].reshape(int(T/2)), lw=0.5)
plt.savefig("Helmholtz_conservative.png", dpi=100)

plt.figure(5)
plt.clf()
plt.plot(range(int(T/2)), x[0, :, n].reshape(int(T/2)))
plt.plot(range(int(T/2)), x[1, :, n].reshape(int(T/2)))


'''
Dissipative (time reversible) simulation
'''
print("dissipative dynamic simulation")


B = D @ Pi  # drift matrix

# Setting up the OU process
dissipative_process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create process

np.random.seed(2)
x = dissipative_process.exact_simulation(x0, epsilon, T, N)  # run simulation

#update finish time if process finished early
T = x.shape[1]
print(f'T= {T}')


'''
Plot trajectory dissipative simulation
'''
print("plotting dissipative dynamic")


plt.figure(6)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)
plt.suptitle('Time-reversible part',fontsize=16)
plt.title(r'$dx_t = b_{rev}(x_t)dt+ \sigma(x_t)dw_t$',fontsize=14)
OU.plot_cool_colourline3(x[0, :, n].reshape(T), x[1, :, n].reshape(T), lw=0.5)
plt.savefig("Helmholtz_dissipative.png", dpi=100)


plt.figure(7)
plt.clf()
plt.plot(range(T), x[0, :, n].reshape(T))
plt.plot(range(T), x[1, :, n].reshape(T))



'''
Plot steady-state density
'''

rv_eye = multivariate_normal(cov= np.eye(dim)) #random normal

lim_x = 2.5
lim_y = 2.5

x_tick = np.linspace(-lim_x, lim_x, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

plt.figure(8)
plt.clf()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, rv_eye.pdf(pos), rstride=1, cstride=1,cmap=cmap_Gaussian, edgecolor='none')
ax.grid(False)
ax.set_zlim(0,1.5*np.max(rv_eye.pdf(pos)))
ax.elev +=-15
ax.axis('off')
ratio = 1.3
len = 8
ax.figure.set_size_inches(ratio * len, len, forward=True)
plt.savefig("3Dplot_Gaussian_density.png", dpi=100)

