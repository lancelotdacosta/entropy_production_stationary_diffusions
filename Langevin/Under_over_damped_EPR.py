'''
EPR of underdamped and overdamped Langevin dynamics in a quadratic potential as a function of inverse temperature
'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import OU_process_functions as OU

'''
MAIN
'''

N = 10 ** 5  # number of trajectories (used to estimate the law of the process via histogram method)
T = 10 ** 4  # number of timesteps to run the process over

l = 1  # lambda = spectral gap = slope of potential
d = 2  # dimension
gamma = 1  # friction param for underdamped
epsilon = 0.01  # learning rate
bins = 3

n = 7  # number of simulations
binvs = np.linspace(1, 10, n) #list of beta inverses (inverse temperatures)

'''
Figure 50. EPR underdamped as a function of friction via inf EPR 
'''
seed = 1
np.random.seed(seed)

epr_underdamped = np.empty([n])  # epr Langevin dynamics
epr_underdamped_median = np.empty([n])  # epr via median estimator

H_under = -np.ones([T, n])  # score entropy
epr_inst_under = -np.ones([T, n])  # instantaneous entropy production rate

i = 0  # set counter
for beta_inv in binvs:
    print(np.round(i / n * 100, 2))  # percentage of simulations done
    B = np.zeros([2 * d, 2 * d])  # drift
    B[:d, d:] = np.eye(d)  # upper right quadrant
    B[d:, :d] = -np.eye(d) * l **2  # lower left quadrant
    B = B / gamma
    sigma = np.zeros([2 * d, 2 * d])  # volatility
    sigma[d:, d:] = np.sqrt(2 * beta_inv) * np.eye(d)  # lower right quadrant
    process = OU.OU_process(dim=2 * d, friction=B,
                            volatility=sigma)  # underdamped Langevin dynamics on quadratic potential
    x = np.zeros([2 * d, 2, N])
    t = 0
    while t < T:
        steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        epr_inst_under[t:(t + steps - 1), i] = OU.inst_epr(x, epsilon,
                                                           nbins=bins)  # record instantaneous entropy production
        H_under[t:(t + steps), i] = OU.entropy(x, nbins=bins)  # record entropy
        t += steps
    epr_underdamped[i] = np.mean(epr_inst_under[epr_inst_under[:, i] >= 0, i])  # score epr
    epr_underdamped_median[i] = np.median(
        epr_inst_under[epr_inst_under[:, i] >= 0, i])  # score epr via median estimator
    i += 1

'''
Plot epr
'''
plt.figure(50)
plt.clf()
plt.plot(binvs, epr_underdamped)

plt.suptitle("Underdamped Entropy production rate (via instantaneous)")

plt.xlabel('beta_inv')
plt.ylabel('epr')

plt.savefig("50.Underdamped.epr(beta).via_instantaneous.png")

'''
Plot epr median
'''
plt.figure(51)
plt.clf()
plt.plot(binvs, epr_underdamped_median)

plt.suptitle("Entropy production rate (via instantaneous), median")

plt.xlabel('beta_inv')
plt.ylabel('epr')

plt.savefig("51.Underdamped.epr(beta).via_instantaneous.median.png")

'''Plot entropy'''
plt.figure(52)
plt.clf()

plt.suptitle("Entropy")

plt.xlabel('time')
plt.ylabel('H')
for i in range(n):
    plt.plot(H_under[:, i])

''' PLot instantaneous EPR'''
plt.figure(53)
plt.clf()
plt.suptitle("Instantaneous epr")

plt.xlabel('time')
plt.ylabel('Inst epr')
for i in range(n):
    plt.plot(epr_inst_under[epr_inst_under[:, i] >= 0, i])

'''
Figure 40. EPR overdamped as a function of friction via inf EPR 
'''
seed = 1
np.random.seed(seed)

epr_overdamped = np.empty([n])  # epr overdamped Langevin dynamics
epr_overdamped_median = np.empty([n])  # epr via median estimator

H_over = -np.ones([T, n])  # score entropy
epr_inst_over = -np.ones([T, n])  # instantaneous entropy production rate

i = 0  # set counter
for beta_inv in binvs:
    print(np.round(i / n * 100, 2))  # percentage of simulations done
    B = - np.eye(d) * l ** 2  # lower left quadrant
    sigma = np.sqrt(2 * beta_inv) * np.eye(d)
    process = OU.OU_process(dim=d, friction=B, volatility=sigma)  # Langevin dynamics on quadratic potential
    x = np.zeros([d, 2, N])
    t = 0
    while t < T:
        steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        epr_inst_over[t:(t + steps - 1), i] = OU.inst_epr(x, epsilon,
                                                          nbins=bins)  # record instantaneous entropy production
        H_over[t:(t + steps), i] = OU.entropy(x, nbins=bins)  # record entropy
        t += steps
    epr_overdamped[i] = np.mean(epr_inst_over[epr_inst_over[:, i] >= 0, i])  # score epr
    epr_overdamped_median[i] = np.median(epr_inst_over[epr_inst_over[:, i] >= 0, i])  # score epr via median estimator
    i += 1

'''
Plot epr
'''
plt.figure(40)
plt.clf()
plt.plot(binvs, epr_overdamped)

plt.suptitle("Overdamped Entropy production rate (via instantaneous)")

plt.xlabel('beta_inv')
plt.ylabel('epr')


plt.savefig("40.Overdamped.epr(beta).via_instantaneous.png")

''' Plot epr median'''
plt.figure(41)
plt.clf()
plt.plot(binvs, epr_overdamped_median)

plt.suptitle("Overdamped Entropy production rate (via instantaneous), median")

plt.xlabel('beta_inv')
plt.ylabel('epr')


plt.savefig("41.Overdamped.epr(beta).via_instantaneous.median.png")

'''Plot entropy'''
plt.figure(42)
plt.clf()

plt.suptitle("Entropy")

plt.xlabel('time')
plt.ylabel('H')
for i in range(n):
    plt.plot(H_over[:, i])

''' PLot instantaneous EPR'''
plt.figure(43)
plt.clf()
plt.suptitle("Instantaneous epr")

plt.xlabel('time')
plt.ylabel('Inst epr')
for i in range(n):
    plt.plot(epr_inst_over[epr_inst_over[:, i] >= 0, i])
