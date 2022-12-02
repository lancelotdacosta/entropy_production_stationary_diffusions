'''
We perform an exact simulation of the 3D OU process for b NOT in Im sigma
We check numerically that the transition kernels as time goes forward or backward are not equivalent
We check that the ep as a function of the time-step tends to infinity as the time-step goes to zero
(using formula for KL of Gaussians and exact simulation)
'''

from OU_Process.Functions import OU_process_functions as OU
import numpy as np
from numpy.linalg import inv, det, pinv, matrix_rank, norm
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
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

dim = 3  # dimension of state-space

std = 1  # define standard deviations of distributions we are sampling random numbers

# Define precision
# Pi = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]) #random precision matrix
# Pi = np.array([3, 1, 0, 1, 3, 0.5, 0, 0.5, 2.5]).reshape([dim, dim]) #selected precision matrix
Pi = np.eye(dim)

# We compute the stationary covariance
S = np.linalg.inv(Pi)

'''
Setting up the OU process
'''

# solenoidal flow
# Q = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary solenoidal flow
# Q = np.zeros([dim, dim])  # no solenoidal flow
Q = np.array([0, 0, 1, 0, 0, 0, -1, 0, 0]).reshape([dim, dim])  # selected solenoidal flow
Q = (Q - Q.T) / 2

# volatility
# sigma = np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim])  # arbitrary volatility matrix
# sigma = np.zeros([dim,dim]) #no noise
# sigma = np.diag(np.random.normal(scale=std, size=dim)) # arbitrary diagonal volatility matrix
sigma = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape([dim, dim])  # @ np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]) #selected non-degenerate noise
# sigma = np.array([2, 1.5, 0.5, 0, 0, 2, 0, 0, 2]).reshape([dim, dim]) #selected degenerate noise

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
    print('h')
    #raise TypeError("Sylvester equation not solved")
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
    for i in range(B_eig.shape[1]): #for each eigenvector index
        if np.all(np.abs(D @ B_eig[:, i]) < tol): #if KRC is false
            return False
    return True


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
x = process.exact_simulation(x0, epsilon, T, N)
#x_rev = rev_process.exact_simulation(x0, epsilon, T, N)

'''
FIGURE 0: Plotting sample trajectory
'''

fig = plt.figure(0)
plt.clf()
ax = plt.axes(projection='3d')
cmap = plt.cm.get_cmap('cool')
cmap_rev = plt.cm.get_cmap('plasma')
for t in range(1, T):
    if t %(2*10**3) ==0:
        print(t)
    ax.plot3D(x[0, t-1:t+1, 0], x[1, t-1:t+1, 0], x[2, t-1:t+1, 0], c=cmap(1-t/T), lw = 0.3)
    #ax.plot3D(x[0, t-1:t+1, 1], x[1, t-1:t+1, 1], x[2, t-1:t+1, 1], c=cmap_rev(1 - t / T), lw=0.3)
ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red', s =20)
ax.text(x0[0], x0[1], x0[2],s='$\mathbf{x_0}$')
plt.title('Sample trajectory \n $b_{\mathrm{irr}} \\notin Im \sigma$')
#ax.legend() #loc="upper right"
plt.savefig("sample_trajectory_3DOU_bnotinImsigma_deg.png")



'''
Setting up the simulation from zero
'''
# this stationary simulation is super important to see how well the synchronisation map works for the stationary process
# despite errors in numerical discretisation and matrix ill-conditioning
# all subsequent simulations can only be trusted to the extent that the synchronisation map
# works in the steady-state simulation

T = 2  # number of time-steps
N = int(10 ** 2)  # number of trajectories
epsilon = 0.01

#simulation
x = process.exact_simulation(x0, epsilon, T, N)
x_rev = rev_process.exact_simulation(x0, epsilon, T, N)


# select the samples of the transition kernel
x = x[:, 1, :].reshape([dim, N])
x_rev = x_rev[:, 1, :].reshape([dim, N])


'''
FIGURE 2: Plotting samples of transition kernels
'''

# plotting samples from the transition kernels
fig = plt.figure(2)
plt.clf()
ax = plt.axes(projection='3d')
plt.title('Transition kernels $b_{\mathrm{irr}} \\notin Im \sigma$\n')
# ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red',label='Starting point')
ax.scatter(x[0, :], x[1, :], x[2, :], marker='o', c='darkorange', label='Forward samples $\sim p_\epsilon(\cdot, x_0)$')
ax.scatter(x_rev[0, :], x_rev[1, :], x_rev[2, :], marker='o', c='cornflowerblue',
           label=r'Backward samples $\sim \bar{p}_\epsilon(\cdot, x_0)$')


'''
Plotting principal components
'''
x_tot = np.concatenate((x, x_rev), axis=1)
pca = PCA(n_components=dim) #start PCA
pca.fit(x_tot.T) #fit PCA to samples

for i in range(dim):
    vector = pca.components_[i,:] #principal component
    length = pca.explained_variance_[i] #explained variance
    v = np.sqrt(length)*pca.components_[i,:]#/(10**i)
    if np.abs(length) > 10 ** (-10):
        if i==0:
            ax.quiver(
                x0[0], x0[1], x0[2],  # <-- starting point of vector
                v[0], v[1], v[2],  # <-- directions of vector
                color='red', alpha=.6, lw=2, arrow_length_ratio=0.03,
                label ='Principal components')
        else:
            ax.quiver(
                x0[0], x0[1], x0[2],  # <-- starting point of vector
                v[0], v[1], v[2],  # <-- directions of vector
                color='red', alpha=.6, lw=2,arrow_length_ratio=0.2)
ax.view_init(elev=25, azim=-36)
ax.legend(loc="upper right")
ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red', s =20)
ax.text(x0[0], x0[1], x0[2],s='$\mathbf{x_0}$')
plt.savefig("transition_kernels_3DOU_bnotinImsigma_deg.png")


'''
FIGURE 3: Plotting ep(epsilon) as a function of epsilon
'''

'''
Computing ep for different timesteps
'''

def pseudodet(A):
    eig_values = np.linalg.eig(A)[0]
    return np.product(eig_values[eig_values > 1e-12])


no_timesteps = 20
epsilons = np.flip(np.geomspace(0.001, 1, no_timesteps))
ep_simul = np.zeros([no_timesteps])  # estimated entropy production rate for several time-steps

for k in range(no_timesteps):
    e = epsilons[k]  # time-step

    # forward process
    exp_B = scipy.linalg.expm(- e * B)
    Q_eps = np.empty([dim, dim])
    for i in range(dim):
        for j in range(dim):
            Q_eps[i, j] = scipy.integrate.quad(func=OU.integrand, a=0, b=e, args=(B, sigma, i, j))[0]
    # backward process
    exp_C = scipy.linalg.expm(- e * C)
    rQ_eps = np.empty([dim, dim])
    for i in range(dim):
        for j in range(dim):
            rQ_eps[i, j] = scipy.integrate.quad(func=OU.integrand, a=0, b=e, args=(C, sigma, i, j))[0]

    ep1 = np.trace(pinv(rQ_eps / e) @ (Q_eps / e)) - matrix_rank(sigma)
    ep2 = np.log(pseudodet(rQ_eps / e) / pseudodet(Q_eps / e))
    ep3 = np.trace(S @ (exp_C/ e - exp_B/ e).T @ pinv(rQ_eps / e) @ (exp_C/e - exp_B/e))
    ep_simul[k] = ((ep1 + ep2)/e + ep3) / 2
    #print(f'timestep={e}, EPR={np.round(ep_simul[k], 1)}')


'''
Plotting ep(epsilon) as a function of epsilon
'''
if matrix_rank(np.concatenate((B,sigma))) > matrix_rank(sigma):
    ep_theo = np.tile(np.infty,no_timesteps)
else:
    ep_theo = np.tile(-np.trace(pinv(D) @ B @ Q), no_timesteps)  # theoretical value of ep

plt.figure(3)
plt.clf()
if ep_theo[0] != np.infty:
    plt.plot(epsilons, ep_theo, c='black', label='$e_p$ from theory', linestyle='dashed')
plt.plot(epsilons, ep_simul, label=f'$e_p(\epsilon)$ from exact simulation')
plt.legend()
plt.xscale('log')
plt.xlabel('Time-step $\epsilon$')
plt.title('Entropy production rate \n $b_{\mathrm{irr}} \\notin Im \sigma$')
# plt.yscale('log')

plt.savefig("ep_timestep_3DOU_bnotinImsigma_deg.png")

print(ep_simul)



'''
FIGURE 4: Plotting deterministic trajectory (if there were no noise)
'''
epsilon = 0.001
fig = plt.figure(4)
plt.clf()
ax = plt.axes(projection='3d')
cmap = plt.cm.get_cmap('cool')
for t in range(1, 2*10**4):
    if t %(2*10**3) ==0:
        print(t)
    y = scipy.linalg.expm(- t*e * B)@x0
    ax.scatter(y[0], y[1], y[2], c='blue')
    #ax.plot3D(x_rev[0, t - 1:t + 1, 0], x_rev[1, t - 1:t + 1, 0], x_rev[2, t - 1:t + 1, 0], c=cmap_rev(1 - t / T), lw=0.3)
ax.scatter(x0[0], x0[1], x0[2], marker='o', c='red', s =20)
ax.text(x0[0], x0[1], x0[2],s='$\mathbf{x_0}$')


'''
Test 5: find generalised Helmholtz decomposition
'''

#find covariance of steady-state (i.e., covariance of forward transition kernel as t-> infty)
Q_infty = np.empty([dim, dim])
for i in range(dim):
    for j in range(dim):
        Q_infty[i, j] = scipy.integrate.quad(func=OU.integrand, a=0, b=np.inf, args=(B, sigma, i, j))[0]
Q_infty = np.round(Q_infty, 2)
