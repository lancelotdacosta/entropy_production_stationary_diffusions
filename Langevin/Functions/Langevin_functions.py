'''
Functions related to Langevin processes
'''

'''
Imports
'''
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
import OU_Process.Functions.OU_process_functions as ou

'''
Process
'''


class Langevin_process(object):

    def __init__(self, dim, gradV, gamma, beta_inv):
        super(Langevin_process, self).__init__()  # constructs the object instance
        self.d = dim
        #TODO: find a way to verify that gradV and V are indeed compatible (automatic differentiation?)
        if np.shape(gradV(np.zeros(self.d))) != np.shape(np.empty(self.d)):
            raise TypeError('error in dimensions gradV')
        self.gradV = gradV
        if gamma <= 0:
            raise TypeError('gamma non-positive')
        self.gamma = gamma  # friction
        if beta_inv <= 0:
            raise TypeError('beta inv non-positive')
        self.beta_inv = beta_inv  # inv temp

    def simulation(self, q0, p0 = 0, epsilon=0.01, T=100, N=1):  # run OU process for multiple trajectories
        w = np.random.normal(0, np.sqrt(epsilon), (T - 1) * self.d * N).reshape(
            [self.d, T - 1, N])  # random fluctuations

        ''' Initialise positions'''
        q = np.empty([self.d, T, N])  # store positions of the process
        if q0.shape == q[:, 0, 0].shape:
            q[:, 0, :] = np.tile(q0, N).reshape([self.d, N])  # initial position
        elif q0.shape == q[:, 0, :].shape:
            q[:, 0, :] = q0
        else:
            raise TypeError("Initial position has wrong dimensions")

        '''Initialise momenta'''
        p = np.empty([self.d, T, N])  # store momenta of the process
        if p0.shape == p[:, 0, 0].shape:
            p[:, 0, :] = np.tile(p0, N).reshape([self.d, N])  # initial position
        elif p0.shape == p[:, 0, :].shape:
            p[:, 0, :] = p0
        else:
            p[:, 0, :] = 0
            print("Initial momenta set to zero")

        '''Run simulation'''
        for t in range(1,T):  # may want to change the update rule with something that is better for Hamiltonian dynamics
            # e.g. book by Lelievre
            p[:, t, :] = (1 - epsilon * self.gamma) * p[:, t - 1, :] \
                         - epsilon * self.gradV(q[:, t - 1, :]) \
                         + np.sqrt(2 * self.beta_inv * self.gamma) * w[:, t - 1, :]  # update momenta
            q[:, t, :] = q[:, t - 1, :] + epsilon * p[:, t, :]  # update positions
            if np.count_nonzero(np.isnan(q)):
                print(f'Time ={t}')
                raise TypeError("nan q")
            if np.count_nonzero(np.isnan(p)):
                print(f'Time ={t}')
                raise TypeError("nan p")
        return q, p

    def attributes(self):
        return [self.d, self.gradV, self.gamma, self.beta_inv]


'''
Functions
'''


def inst_epr(q, p, epsilon, nbins=10):  # compute instantaneous EPR
    return ou.inst_epr(np.append(q, p, axis=0), epsilon, nbins)


def epr_via_inst(process, N, T, epsilon, steps=10, bins=2, mc=True):
    d = process.attributes()[0]
    q = np.zeros([d,1,N]) #initialise positions at zero
    p = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d), size=[N,1]).T #initialise momenta at steady state
    t = 0
    epr_inst = -np.ones([T])  # instantaneous entropy production rate
    # TODO: can augment this by recording only at steady state: ie when entropy doesn't increase much
    while t < T:
        q, p = process.simulation(q[:, -1, :], p[:, -1, :], epsilon, T=steps, N=N)
        epr_inst[t:(t + steps - 1)] = inst_epr(q, p, epsilon,
                                               nbins=bins)  # record instantaneous entropy production
        t += steps
    epr_v = np.mean(epr_inst[epr_inst >= 0])
    epr_v_median = np.median(epr_inst[epr_inst >= 0])
    if mc:
        epr_mc = epr_int_MC(process, q, p)
        return epr_v, epr_v_median, epr_mc
    else:
        return epr_v, epr_v_median


def epr_int_MC(process, q, p):  # integral formula for EPR
    [T, N] = q.shape[1:]
    gradV, gamma, beta_inv = process.attributes()[1:]

    '''Estimate entropy production'''
    e_p = 0
    for i in range(T):
        for j in range(N):
            # TODO: check that the computation is correct and write on latex
            y1 = -gradV(q[:,i,j])+gamma*(beta_inv-1)*p[:,i,j]
            e_p += np.dot(y1,y1)/(beta_inv*gamma*N)
    return e_p/T


'''Auxiliary functions'''

def bin_range(x):
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]), np.max(x[d, :, :])])
    return b_range
