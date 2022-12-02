'''
We verify the expression for partial_e Tr(bar Q_e^- Q_e) at e=0
for an OU process whose Image of the probability current is included in the image of the volatility
'''

import autograd.numpy as np
from autograd import grad
from numpy.linalg import inv, det, eig, pinv
from numpy.linalg import eigvals as spec
from numpy.linalg import matrix_rank as rank
from numpy import trace
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import OU_Process.Functions.OU_process_functions as ou

dim = 2 * np.random.choice(np.arange(1, 3, 1)) + 1  # dimension of state-space
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
Solenoidal flow
'''
# solenoidal flow
Q = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)  # arbitrary solenoidal flow
Q = Q - Q.T

print(f'rank Q = {rank(Q)}')

'''
Volatility
'''

# set up invertible matrix
arb = np.zeros([dim, dim])
while np.abs(det(arb)) < 0.1:
    arb = np.round(np.random.normal(scale=std, size=dim ** 2).reshape([dim, dim]), 1)

# set up volatility matrix as Q times invertible matrix
sigma = Q @ arb

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
print(f'rank sigma = {rank(sigma)}')
print(f'dim = {dim}, KRC forward = {KRC_f}, KRC backward = {KRC_b}')

'''
Functions
'''


def integrand(t, B, sigma, i, j):
    mx = scipy.linalg.expm(-B * t) @ sigma
    mx = mx @ mx.T
    return mx[i, j]


def auxQ(B, e):  # compute Q for given drift B and epsilon e
    integral = np.empty([dim, dim])
    for i in range(dim):
        for j in range(dim):
            integral[i, j] = scipy.integrate.quad(func=integrand, a=0, b=e, args=(B, sigma, i, j))[0]
    return integral


#def Q(e):  # forward Q
#    return auxQ(Bf, e)


def bQ(e):  # backward Q
    return auxQ(Bb, e)


#def g(e):
#    return trace(pinv(bQ(e)/e) @ Q(e)/e) - rank(sigma)


'''
EPR1: We compute the formula for the derivative at 0 of g
'''
I = np.eye(dim)
a1 = pinv(D) @ (Bb @ D + D @ Bb.T) @ pinv(D) @ D
a2 = -pinv(D) @ pinv(D) @ (Bb @ D + D @ Bb.T) @ (I - D @ pinv(D)) @ D
a3 = -(I - pinv(D) @ D) @ (Bb @ D + D @ Bb.T) @ pinv(D) @ pinv(D) @ D
a4 = - pinv(D) @ (Bf @ D + D @ Bf.T)
formula1 = trace(a1 + a2 + a3 + a4) / 2



'''
EPR2:
'''
J= (Bf-Bb)/2
b1 = pinv(D)@(J@D+D@J.T)@(2*I-pinv(D)@D)
b2= pinv(D)@pinv(D)@(J@D+D@J.T)@(I-D@pinv(D))@D
b3= (I-pinv(D)@D)@(J@D+D@J.T)@pinv(D)@pinv(D)@D
formula2= trace(b1 + b2 + b3)


'''
Misc
'''

print(trace(Q @pinv(D)@D@ Pi))
