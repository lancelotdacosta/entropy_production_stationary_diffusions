'''
Computation of weak derivative of paths in Cameron-Martin space
as time goes forward and backward for a 2D OU process with degenerate noise
#following computations from 18 Jun 2021
'''

import numpy as np
from numpy.linalg import inv, det, eig
from numpy.linalg import eigvals as spec
from numpy.linalg import matrix_rank as rank
from scipy.linalg import expm
from SubRoutines.Auxiliary import num
import matplotlib.pyplot as plt

dim = 2  # dimension of state-space
# np.random.seed(1)
std = 1  # define standard deviations of distributions we are sampling random numbers

'''
Precision matrix
'''

Pi = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)  # random precision matrix
# enforce symmetric
Pi = (Pi + Pi.T) / 2
# make sure Pi is positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)

'''
Volatility
'''
# setup arbitrary change of basis matrix
arb = np.zeros([dim, dim])
while np.abs(det(arb)) < 0.1:
    arb = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)

# setup eigenvalues of sigma
n_nonzero_eigs = np.random.choice(np.arange(1, dim, 1))  # number of non-zero eigenvalues in sigma (between 1 and dim-1)
random_eigs = np.round(np.random.normal(scale=std, size=n_nonzero_eigs), 1)
random_eigs = random_eigs + 2 * (random_eigs >= 0) - 1
deg = np.zeros([dim, dim])
deg[range(n_nonzero_eigs), range(n_nonzero_eigs)] = random_eigs

# set up volatility matrix as change of basis of the deg matrix
sigma = inv(arb) @ deg @ arb

# check whether noise is degenerate or not
print(f'rank sigma = {rank(sigma)}, dim = {dim}')
if rank(sigma) != n_nonzero_eigs:  # sanity check
    print('error')

# diffusion tensor
D = sigma @ sigma.T / 2  # diffusion tensor

'''
Solenoidal flow
'''
# solenoidal flow
Q = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)  # arbitrary solenoidal flow
# Q = np.zeros([dim, dim])  # no solenoidal flow
# Q = np.array([0,1,0,0]).reshape([dim, dim])
Q = (Q - Q.T) / 2

'''
Drift matrices (as time goes forward and backward)
'''
Bf = (D + Q) @ Pi  # drift as time goes forward
Bb = (D - Q) @ Pi  # drift as time goes backward

'''
Kalman rank condition
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


KRC_f = KRC(Bf, D)  # Kalman rank condition as time goes forward
KRC_b = KRC(Bf, D)  # Kalman rank condition as time goes backward
print(f'KRC forward = {KRC_f}, KRC backward = {KRC_b}')

'''
Simulation
'''
T = np.arange(0, dim + 1, 1)
expBf = np.empty([num(T), dim, dim])
expBb = np.empty([num(T), dim, dim])
Im_f = np.empty([num(T), dim, dim])
Im_b = np.empty([num(T), dim, dim])

for t in T:
    expBf[t, :, :] = expm(Bf * t)  # exponential drift matrix as time goes forward
    expBb[t, :, :] = expm(Bb * t)  # exponential drift matrix as time goes backward
    for i in range(dim):
        e_i = np.zeros([dim])
        e_i[i] = 1  # canonical basis vector
        Im_f[t, i, :] = expBf[t] @ sigma @ e_i
        Im_b[t, i, :] = expBb[t] @ sigma @ e_i

    if t > 0:
        print(
            f't= {t}: slope forward = {Im_f[t, 0, 1] / Im_f[t, 0, 0]}, slope backward = {Im_b[t, 0, 1] / Im_b[t, 0, 0]}')

        '''
        Combined plot
        '''
        plt.figure(t)
        plt.clf()
        plt.suptitle(f'Time = {t}')
        plt.scatter(Im_f[t, :, 0], Im_f[t, :, 1], alpha=0.5, c='royalblue', label='Forward time')
        plt.scatter(Im_b[t, :, 0], Im_b[t, :, 1], alpha=0.5, c='darkorange', label='Backward time')
        origin = np.zeros([dim])
        plt.quiver(origin, origin, Im_f[t, :, 0], Im_f[t, :, 1], alpha=0.5, color='royalblue', scale=1)
        plt.quiver(origin, origin, Im_b[t, :, 0], Im_b[t, :, 1], alpha=0.5, color='darkorange', scale=1)
        plt.legend(framealpha=0.5)
