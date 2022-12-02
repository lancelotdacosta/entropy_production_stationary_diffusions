'''
Illustration Helmholtz decomposition 2 way OU process
'''

import OU_Process.Functions.OU_process_functions as OU
import numpy as np
from numpy.linalg import inv, det,pinv
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.rcParams["text.usetex"] =True
plt.style.use('seaborn-white')
from scipy.stats import multivariate_normal
import seaborn as sns


np.random.seed(3)
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
sigma[0,0]=1
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
Q = (Q - Q.T)

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





'''
OU process sample paths
'''

epsilon = 10**(-4)  # time-step
T = 2*10 ** 4 #10 ** 4  # number of time-steps
#T = 5*10**4
N = 2  # number of trajectories
n = 1  # which sample path to show (between 0 and N-1)


# Setting up the initial condition
#x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=[N]).reshape([dim, N])  # stationary initial condition
x0 = np.ones(dim)/3

#simulation counter
i=0

for gamma in [0,1,10]:

    # sample paths
    #np.random.seed(2)
    # Setting up the OU process
    print(f'starting simulation gamma={gamma}')
    B = (D + gamma * Q) @ Pi  # drift matrix
    process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create process
    x = process.exact_simulation(x0, epsilon, T, N)  # run simulation
    #print(f'x-axis: {np.max(x[n,:,0])} to {np.min(x[n,:,0])}')
    #print(f'y-axis: {np.max(x[n,:,1])} to {np.min(x[n,:,1])}')


    #update finish time if process finished early
    #T = x.shape[1]
    #print(f'T= {T}')

    '''
    Plotting sample path 
    '''
    print(f'plotting simulation gamma={gamma}')


    #lim_x = 1.5
    #lim_y = 1.5

    x_tick = np.linspace(-1.2, 1.35, 105)  # x axis points
    y_tick = np.linspace(-2.35, 1.8, 100)  # y axis points
    #x_tick = np.linspace(np.min(x[0, :, n]) - 0.5, np.max(x[0, :, n]) + 0.5, 105)  # x axis points
    #y_tick = np.linspace(np.min(x[1, :, n]) - 0.5, np.max(x[1, :, n]) + 0.5, 100)  # y axis points

    X,Y = np.meshgrid(x_tick,y_tick)
    pos = np.dstack((X, Y))

    rv = multivariate_normal(cov= S) #random normal

    plt.figure(i)
    plt.clf()
    plt.title('')
    plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)  # plotting the free energy landscape

    plt.suptitle(f'Sample trajectory',fontsize=16)
    plt.title(r'$b_{\mathrm{irr}}$ scaling factor $\theta=$'+f' {gamma}',fontsize=14)

    OU.plot_cool_colourline3(x[0, :, n].reshape(T), x[1, :, n].reshape(T), lw=0.5)
    plt.savefig(f"OU2d_sample_path_func_gamma={gamma}.png", dpi=100)


    #plt.figure(i+1)
    #plt.clf()
    #plt.plot(range(T), x[0, :, n].reshape(T))
    #plt.plot(range(T), x[1, :, n].reshape(T))

    i= i+2



'''Plot EPR as a function of irreversible scaling parameter gamma'''
print('Plotting EPR as a function of gamma')

n_params = 1000
gammas =np.linspace(0, 10, 1000)
epr_gamma = np.empty(n_params)
epr_gamma_alter = np.empty(n_params)
for i in range(n_params):
    gamma=np.linspace(0, 10, 1000)[i]
    epr_gamma[i]= -np.trace(pinv(D) @ Q @ Pi @ Q)*gamma**2
    B=(D + gamma * Q) @ Pi
    epr_gamma_alter[i]= -np.trace(pinv(D) @ B @ Q)*gamma

plt.figure(i)
plt.clf()
plt.plot(gammas, epr_gamma, label=r'$e_p(\theta)$')
#plt.plot(gammas, epr_gamma_alter, c='black', linestyle='dashed')
plt.legend()
plt.xlabel(r'$\theta$')
#plt.ylabel(r'$e_p(\gamma)$')
plt.suptitle('Entropy production rate', fontsize=16)
plt.title(r'function of $b_{\mathrm{irr}}$ scaling factor $\theta$', fontsize=14)
# plt.yscale('log')
plt.savefig(f"OU2d_EPR_func_gamma.png", dpi=100)



