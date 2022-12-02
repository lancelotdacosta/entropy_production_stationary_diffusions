'''
24.6.2021
Here we look at what KRC implies for the relationship between Q and sigma
'''


import numpy as np
from numpy.linalg import inv, det, eig
from numpy.linalg import eigvals as spec
from numpy.linalg import matrix_rank as rank
from scipy.linalg import expm
import matplotlib.pyplot as plt

# np.random.seed(1)
std = 1  # define standard deviations of distributions we are sampling random numbers

'''
Dimension
'''

maxdim = 20  # dimension of state-space

dim = np.random.choice(np.arange(2, maxdim + 1, 1))

'''
Precision matrix
'''

Pi = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)  # random precision matrix
# enforce symmetric
Pi = (Pi + Pi.T) / 2
# make sure Pi is positive definite
if np.any(spec(Pi) <= 0):
    Pi = Pi - 2 * np.min(spec(Pi)) * np.eye(dim)


Pi =np.eye(dim)


'''
Solenoidal flow
'''
# solenoidal flow
Q = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)  # arbitrary solenoidal flow
# Q = np.zeros([dim, dim])  # no solenoidal flow
# Q = np.array([0,1,0,0]).reshape([dim, dim])
Q = (Q - Q.T) / 2 #make antisymmetric

'''
Volatility
'''

sigma = np.sqrt(2)*Q

# diffusion tensor
D = sigma @ sigma.T / 2  # diffusion tensor

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
print(f'rankQ={rank(Q)}')
print(f'dim = {dim}, KRC forward = {KRC_f}, KRC backward = {KRC_b}')