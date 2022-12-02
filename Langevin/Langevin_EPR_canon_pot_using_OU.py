'''
EPR of Langevin dynamics in a quadratic potential
'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import OU_Process.Functions.OU_process_functions as OU

'''
MAIN
'''

N = 10 ** 3  # number of trajectories (used to estimate the law of the process via histogram method)
T = 10 ** 4  # number of timesteps to run the process over
bins = 2

n = 10  # number of epr computations and friction coefficients
gammas = np.flip(np.linspace(10**(-2), 205, n))  # friction param; small=underdamped, large=overdamped

'''
Figure 60. EPR Langevin as a function of friction via inf EPR 
'''
seed = 1
np.random.seed(seed)

d = 2  # dimension
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step
steps = 10  # number of steps in a simulation

epr_Langevin = np.empty([n])  # epr Langevin dynamics
epr_Langevin_median = np.empty([n])  # epr via median estimator

#H = -np.ones([T, n])  # score entropy
epr_inst = -np.ones([int(T/steps)+1, n])  # instantaneous entropy production rate

l = 1  # lambda = spectral gap = slope of potential
beta_inv = 1  # inverse temperature

'''Careful: this simulation does not launch samples at steady-state'''

for i in range(n):
    print(f'{np.round(i / n * 100, 2)}%')  # percentage of simulations done
    B = np.zeros([2 * d, 2 * d])  # drift
    B[:d, d:] = np.eye(d)  # upper right quadrant
    B[d:, :d] = -np.eye(d) * l ** 2  # lower left quadrant
    B[d:, d:] = -np.eye(d) * gammas[i]  # lower right quadrant
    sigma = np.zeros([2 * d, 2 * d])  # volatility
    sigma[d:, d:] = np.sqrt(2 * gammas[i] * beta_inv) * np.eye(d)  # lower right quadrant
    process = OU.OU_process(dim=2*d, friction=-B, volatility=sigma)  # Langevin dynamics on quadratic potential
    x = np.zeros([2 * d, 2, N])
    t = 0
    while t < T:
        x = process.simulation_float128(x[:, -1, :], epsilon, T=steps, N=N)
        if not np.all(np.isfinite(x)):
            raise TypeError('error in x')
        epr_inst[int(t/steps), i] = np.mean(OU.inst_epr(x, epsilon, nbins=bins))  # record instantaneous entropy production
       # H[t:(t + steps), i] = OU.entropy(x, nbins=bins)  # record entropy
        t += steps
    epr_Langevin[i] = np.mean(epr_inst[epr_inst[:, i] >= 0, i])  # score epr
    epr_Langevin_median[i] = np.median(epr_inst[epr_inst[:, i] >= 0, i])  # score epr via median estimator


'''
Plot epr
'''
plt.figure(62)
plt.clf()
plt.xlabel('friction gamma')
plt.ylabel('epr')
plt.xscale('log')
plt.plot(gammas, epr_Langevin)

plt.suptitle("Langevin Entropy production rate (via instantaneous)")

plt.savefig("60.Langevin.epr(gamma).via_instantaneous.png")

'''
Plot epr median
'''
plt.figure(63)
plt.clf()
plt.plot(gammas, epr_Langevin_median)

plt.suptitle("Langevin Entropy production rate (via instantaneous), median")

plt.xlabel('gamma')
plt.ylabel('epr')
plt.xscale('log')

plt.savefig("61.Langevin.epr(gamma).via_instantaneous.median.png")

'''Plot entropy'''
plt.figure(64)
plt.clf()

plt.suptitle("Entropy")

plt.xlabel('time')
plt.ylabel('H')
#for i in range(n):
    # plt.plot(H[:, i])

''' Plot instantaneous EPR'''
plt.figure(65)
plt.clf()
plt.suptitle("Instantaneous epr")

plt.xlabel('time')
plt.ylabel('Inst epr')
for i in range(n):
    plt.plot(epr_inst[epr_inst[:, i] >= 0, i])
