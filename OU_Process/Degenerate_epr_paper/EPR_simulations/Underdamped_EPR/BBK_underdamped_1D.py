'''
We perform a BBK simulation of the 1D underdamped Langevin dynamics in a quadratic potential
We check numerically that the transition kernels as time goes forward or backward are equivalent
We estimate ep as a function of the time-step
'''

from OU_Process.Functions import OU_process_functions as OU
import numpy as np
from numpy.linalg import inv, det, pinv, matrix_rank, norm
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from scipy.stats import multivariate_normal
import scipy
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

pgf_with_latex = {"pgf.preamble": r"\usepackage{amsmath}"}  # setup matplotlib to use latex for output
plt.style.use('seaborn-white')

np.random.seed(1)
'''
Setting up the steady-state
'''

dim = 2  # dimension of state-space


# Define precision
Pi = np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)

'''
Setting up the underdamped Langevin dynamics
'''
#friction=mass =1
#inverse temperature = beta =1
#nabla potential(q) = q
#in other words, potential V(q)=q^2/2

# solenoidal flow
Q = np.array([0, -1, 1, 0]).reshape([dim, dim])  # selected solenoidal flow

# volatility
sigma = np.array([0, 0, 0, np.sqrt(2)]).reshape([dim, dim])  # @ np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]) #selected non-degenerate noise

# see whether noise is degenerate or not
print(f'det sigma = {det(sigma)}')

# diffusion tensor
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

# Reverse drift
C = (D - Q) @ Pi

# Setting up the OU process
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)  # create forward process
rev_process = OU.OU_process(dim=dim, friction=C, volatility=sigma)  # create reverse process



'''
Kalman rank condition (i.e., parabolic Hormander's condition)
'''
# we verify whether Kalman rank condition holds


def KRC(B, D):  # test Kalman rank condition
    # KRC holds iff forall v eigenvector of B.T we have Dv != 0
    # KRC doesn't hold iff exists v eigenvector of B.T with Dv==0
    B_eig = np.linalg.eig(B.T)[1]
    tol = 10 ** (-6)
    KRC = True
    for i in range(B_eig.shape[1]):
        if np.all(np.abs(D @ B_eig[:, i]) < tol):
            KRC = False
    return KRC


KRC_f = KRC(B, D)  # Kalman rank condition as time goes forward
KRC_b = KRC(C, D)  # Kalman rank condition as time goes backward
print(f'KRC forward = {KRC_f}, KRC backward = {KRC_b}')

# print(np.linalg.matrix_rank(np.concatenate((B, sigma))))

'''
Setting up the simulation from zero
'''
# this stationary simulation is super important to see how well the synchronisation map works for the stationary process
# despite errors in numerical discretisation and matrix ill-conditioning
# all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

T = 2*10**4  # number of time-steps
N = 2  # number of trajectories
epsilon = 0.001

# Setting up the initial condition to zero
x0 = np.ones([dim])

#doing simulation
x = process.BBK_simulation(x0, epsilon, T, N)
#x_rev = rev_process.exact_simulation(x0, epsilon, T, N)

'''
FIGURE 0: Plotting sample trajectory
'''

fig = plt.figure(1)
plt.clf()
cmap = plt.cm.get_cmap('cool')
#cmap_rev = plt.cm.get_cmap('plasma')
for t in range(1, T):
    if t %(2*10**3) ==0:
        print(t)
    plt.plot(x[0, t-1:t+1, 0], x[1, t-1:t+1, 0], c=cmap(1-t/T), lw = 0.3)
    #ax.plot3D(x_rev[0, t - 1:t + 1, 0], x_rev[1, t - 1:t + 1, 0], x_rev[2, t - 1:t + 1, 0], c=cmap_rev(1 - t / T), lw=0.3)
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
plt.xlabel('q')
plt.ylabel('p')
plt.title('Sample trajectory \n BBK discretisation of underdamped Langevin')
#ax.legend() #loc="upper right"
plt.savefig("sample_trajectory_1D_underdamped_BBK.png")



'''
Simulation of transition kernels
'''
# this stationary simulation is super important to see how well the synchronisation map works for the stationary process
# despite errors in numerical discretisation and matrix ill-conditioning
# all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

T = 2  # number of time-steps
N = int(2.5*10 ** 2)  # number of trajectories
epsilon = 0.01

#simulation
x = process.BBK_simulation(x0, epsilon, T, N)
x_rev = rev_process.exact_simulation(x0, epsilon, T, N)


# select the samples of the transition kernel
x = x[:, 1, :].reshape([dim, N])
x_rev = x_rev[:, 1, :].reshape([dim, N])


'''
FIGURE 2: Plotting samples of transition kernels
'''

#determine the boundaries of the plot
sup_q = 1.015
inf_q = 0.987
sup_p = 1.5
inf_p = 0.38

fig = plt.figure(3)
plt.clf()
plt.title('Transition kernels \n BBK discretisation of underdamped Langevin')
plt.xlabel('q')
plt.ylabel('p')

# plotting heat-map from true forward transition kernel
q_tick = np.linspace(inf_q, sup_q, 105)  # x axis points
p_tick = np.linspace(inf_p, sup_p, 100)  # y axis points
X,Y = np.meshgrid(q_tick,p_tick)
pos = np.dstack((X, Y))

forward_kernel = process.transition_kernel(x0,epsilon)

plt.contourf(X, Y, forward_kernel.pdf(pos), levels=100, cmap='Greys')  # plotting the heat map of transition kernel
black_patch = mpatches.Patch(color='Black', label=r'Exact kernel $p_\epsilon(\cdot, x_0)$')


# plotting samples from Euler forward transition kernel
# ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red',label='Starting point')
plt.scatter(x[0, :], x[1, :], marker='o', c='darkorange', alpha= 0.3)#, label=r'Forward samples $\sim p^{E\mathrm{-}M}_\epsilon(\cdot, x_0)$')
#plt.scatter(x_rev[0, :], x_rev[1, :], marker='o', c='cornflowerblue', label='Backward samples$\sim p^-_\epsilon(\cdot, x_0)$')
#orange_patch = mpatches.Circle(xy= (0,0), color='darkorange', fill=True, label=r'Forward samples $\sim p^{E\mathrm{-}M}_\epsilon(\cdot, x_0)$')
orange_patch = Line2D([0], [0], marker='o', color='w', label=r'Forward samples $\sim p^{BBK}_\epsilon(\cdot, x_0)$',markerfacecolor='darkorange', markersize=8)

#set plot limits
plt.xlim([inf_q, sup_q])
plt.ylim([inf_p, sup_p])

plt.legend(handles=[black_patch, orange_patch], loc="upper center")
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
plt.savefig("transition_kernels_underdamped_1D_BBK.png")





'''
-older


# plotting samples from the transition kernels
fig = plt.figure(4)
plt.clf()
plt.title('Transition kernels \n BBK discretisation of underdamped Langevin')
plt.xlabel('q')
plt.ylabel('p')
# ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red',label='Starting point')
plt.scatter(x[0, :], x[1, :], marker='o', c='darkorange', label='Forward samples$\sim p_\epsilon(\cdot, x_0)$')
plt.scatter(x_rev[0, :], x_rev[1, :], marker='o', c='cornflowerblue',
           label='Backward samples$\sim p^-_\epsilon(\cdot, x_0)$')

plt.legend(loc="upper right")
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
#plt.savefig("transition_kernels_underdamped_1D_BBK.png")
'''







'''
FIGURE 3: Estimating ep(epsilon) of BBK simulation as a function of epsilon
'''

'''
Setting up steady-state simulation
'''
# this stationary simulation is super important to see how well the synchronisation map works for the stationary process
# despite errors in numerical discretisation and matrix ill-conditioning
# all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

T = 10 ** 2  # number of time-steps
N = 10 ** 5  # number of trajectories

# start many trajectories at a really high free energy

# Setting up the initial condition
x0 = np.random.multivariate_normal(mean=np.zeros(dim), cov=S, size=[N]).reshape([dim, N])  # stationary initial condition

no_timesteps = 24
epsilons = np.geomspace(0.001, 0.5, no_timesteps)
no_bins = 3
bins = np.linspace(8, 16, no_bins)
ep_simul = np.empty([no_timesteps, no_bins])  # estimated entropy production rate for several time-steps

'''
Estimating e_p for different time-steps
'''

for i in range(no_timesteps):
    epsilon = epsilons[i]  # time-step
    x = process.BBK_simulation(x0, epsilon, T, N)  # run simulation
    for b in range(no_bins):
        print(f'i={i}, b={b}')
        nbin = int(bins[b])
        # estimate entropy production rate via binning method
        ep_simul[i, b] = OU.epr_samples(x, epsilon, nbins=nbin)
    del x

'''
Plotting ep(epsilon) as a function of epsilon
'''

plt.figure(5)
plt.clf()
for b in range(no_bins):
    plt.plot(epsilons, ep_simul[:,b], label=r'Approx. of $e_p^{BBK}(\epsilon)$,'+f' {int(bins[b])} bins', linestyle='dashed')
plt.legend()
plt.xscale('log')
plt.xlabel('Time-step $\epsilon$')
plt.title('Entropy production rate \n BBK discretisation of underdamped Langevin')
# plt.yscale('log')

plt.savefig("ep_timestep_1D_underdamped_BBK.png")