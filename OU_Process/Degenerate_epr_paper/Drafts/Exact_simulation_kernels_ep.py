'''
DRAFT
We perform an exact simulation of the OU process
We check that the ep as the time-step goes to zero converges to the ep of the true process
We check whether the transition kernels as time goes forward or backward are equivalent
'''


from OU_Process.Functions import OU_process_functions as OU
import numpy as np
from numpy.linalg import inv, det,pinv
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from scipy.stats import multivariate_normal
pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.rcParams["text.usetex"] =True
import matplotlib.cm as cm
import seaborn as sns

np.random.seed(1)
'''
Setting up the steady-state
'''

dim = 2  # dimension of state-space

# Define precision
# Pi = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]) #random precision matrix
#Pi = np.array([3, 1, 1, 2]).reshape([dim, dim]) #selected precision matrix
Pi = np.eye(dim)
# enforce symmetric
Pi = (Pi + Pi.T) / 2
# make sure Pi is positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)


'''
Setting up the OU process
'''

# solenoidal flow
#Q = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary solenoidal flow
# Q = np.zeros([dim, dim])  # no solenoidal flow
Q = np.array([0, 1.5, -1.5, 0]).reshape([dim, dim]) #selected solenoidal flow
Q = (Q - Q.T)/2

# volatility
# sigma = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary volatility matrix
# sigma = np.zeros([dim,dim]) #no noise
# sigma = np.diag(np.random.normal(scale=std, size=dim)) # arbitrary diagonal volatility matrix
sigma = np.array([0.5, 1.5, 0.5, 0]).reshape([dim, dim]) #selected non-degenerate noise
#sigma = np.array([2, 1.5, 0.5, 0, 0, 2, 0, 0, 2]).reshape([dim, dim]) #selected degenerate noise

# see whether noise is degenerate or not
print(f'det sigma = {det(sigma)}')

#diffusion tensor
D = sigma @ sigma.T / 2  # diffusion tensor

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

#Reverse drift
C= (D - Q) @ Pi

# Setting up the OU process
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create forward process
rev_process = OU.OU_process(dim=dim, friction=C, volatility=sigma)  # create reverse process

'''
Setting up the simulation from zero
'''
#this stationary simulation is super important to see how well the synchronisation map works for the stationary process
#despite errors in numerical discretisation and matrix ill-conditioning
#all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

T = 2  # number of time-steps
N = 10 ** 5  # number of trajectories
epsilon = 0.01

# Setting up the initial condition to zero
x0 = np.ones([dim])

x = process.simulation(x0, epsilon, T, N)
x_rev = rev_process.simulation(x0, epsilon, T, N)


'''
Getting transition kernels
'''
#forward process
exp_B = scipy.linalg.expm(- epsilon * B)
Q_eps = np.empty([dim, dim])
for i in range(dim):
    for j in range(dim):
        Q_eps[i, j] = scipy.integrate.quad(func=OU.integrand, a=0, b=epsilon, args=(B, sigma, i, j))[0]
trans = multivariate_normal(exp_B@x0, Q_eps) #transition kernel forward process

#backward process
exp_C = scipy.linalg.expm(- epsilon * C)
rQ_eps = np.empty([dim, dim])
for i in range(dim):
    for j in range(dim):
        rQ_eps[i, j] = scipy.integrate.quad(func=OU.integrand, a=0, b=epsilon, args=(C, sigma, i, j))[0]
rev_trans = multivariate_normal(exp_C@x0, rQ_eps) #transition kernel backward process


'''
Plotting transition kernels for the first timestep
'''

lim_x = 2.5
lim_y = 2.5

x_tick = np.linspace(0.7, 1.3, 105)  # x axis points
y_tick = np.linspace(0.7, 1.3, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

plt.figure(0)
plt.clf()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, trans.pdf(pos), rstride=1, cstride=1, cmap='Reds', edgecolor='none',alpha=0.3) #label='$p_\epsilon(\cdot, x)$'
ax.plot_surface(X, Y, rev_trans.pdf(pos), rstride=1, cstride=1, cmap='Blues', edgecolor='none', alpha=0.3) #label='$p^-_\epsilon(\cdot, x)$'
ax.grid(False)
ax.set_zlim(0,1.3*np.max(trans.pdf(pos)))
#ax.elev +=-15
#ratio = 1.3
#len = 8
ax.legend()
ax.view_init(60, 35)
#ax.figure.set_size_inches(ratio * len, len, forward=True)
plt.savefig("3Dplot_Gaussian_density.png", dpi=100)

'''
#Create grid and multivariate normal
x = np.linspace(0,3,1000)
y = np.linspace(0,3,1000)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y

#Make a 3D plot
fig = plt.figure(0)
plt.clf()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, trans.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
'''
