'''
Overdamped Langevin dynamics on a quadratic potential
'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import OU_process_empirical_EPR as OU

'''
MAIN
'''

seed = 1
np.random.seed(seed)

'''
Figure 40. OLD EPR theoretical
'''
d = 2  # dimension
n = 1  # number of epr computations
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

gradV = np.eye(d)  # symmetric positive matrix
sigma = np.eye(d)  # volatility matrix

epr = np.zeros([n])  # epr
deltas = np.linspace(0, 1 - 0.01, n)
i = 0  # set counter
for delta in deltas:
    process = OU.OU_process(dim=d, friction=gradV, volatility=sigma)
    epr[i] = OU.ent_prod_rate_2D(process)
    i += 1

plt.figure(40)
plt.clf()
OU.plot_cool_colourline2(deltas, epr, deltas, lw=1)

plt.suptitle("Entropy production rate (theoretical)")

plt.xlabel('delta')
plt.ylabel('epr')
plt.savefig("40.OLD.EPR.theoretical.png")
# plt.clf()


'''
Figure 41. EPR via Monte Carlo
'''
print("Starting Figure 41")
N = 10**3  # number of Monte carlo estimates (ie trajectories
T = 10**2  # number of timesteps

D = 0.5 * sigma * sigma.T  # diffusion tensor

epr_mc = np.empty([n])  # epr monte carlo estimate
i = 0  # set counter

for delta in deltas:
    process = OU.OU_process(dim=d, friction=gradV, volatility=sigma)
    x = process.simulation(x0, epsilon, T, N)
    S = process.stat_cov2D()
    if np.linalg.det(S) == 0:
        print(i)
        print(S)
        raise TypeError('Not hypoelliptic')
    if np.linalg.det(D) == 0:
        print(i)
        print(D)
        raise TypeError('Not elliptic')
    Sinv = np.linalg.inv(S)
    Q = gradV @ S - D
    Dinv = np.linalg.inv(D)
    for sim in range(0, N):
        y = Q @ Sinv @ x[:, -1, sim] #probability flux
        epr_mc[i] += y.T @ Dinv @ y #integrand of entropy prod rate
    epr_mc[i] = epr_mc[i] / N
    i += 1

plt.figure(41)
plt.clf()
OU.plot_cool_colourline2(deltas, epr_mc, deltas, lw=1)

plt.suptitle("Entropy production rate (Monte-Carlo)")

plt.xlabel('delta')
plt.ylabel('epr')

plt.savefig("41.OLD.EPR.Monte-Carlo.png")

'''
Figure 42. EPR via instantaneous epr
'''

print("Starting Figure 42")

N = 10 ** 5  # number of trajectories (used to estimate the law of the process via histogram method)
T = 1 * 10 ** 4  # number of timesteps to run the process over

epr_v = np.empty([n])  # epr via instantaneous
epr_v2 = np.empty([n])  # epr via instantaneous median
H = -np.ones([T, n])  # score entropy
epr_inst = -np.ones([T, n])  # instantaneous entropy production rate
# pos = np.zeros([d, T, 1, n])  # positions

i = 0  # set counter
for delta in deltas:
    print(np.round(i / n * 100, 2))
    process = OU.OU_process(dim=d, friction=gradV, volatility=sigma)
    S = process.stat_cov2D()  # stationary covariance
    x = np.transpose(np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N,
                                                                                    1]))  # generate initial condition at steady-state (since known)
    t = 0
    while t < T:
        steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        # pos[:, t:(t + steps), 0, i] = x[:, :, 0]  # record positions
        epr_inst[t:(t + steps - 1), i] = OU.inst_epr(x, epsilon, nbins=10)  # record instantaneous entropy production
        H[t:(t + steps), i] = OU.entropy(x, nbins=10)  # record entropy
        t += steps
    epr_v[i] = np.mean(epr_inst[epr_inst[:, i] >= 0, i])
    epr_v2[i] = np.median(epr_inst[epr_inst[:, i] >= 0, i])
    i += 1

# Plotting epr
plt.figure(42)
plt.clf()
OU.plot_cool_colourline2(deltas, epr_v, deltas, lw=1)

plt.suptitle("Entropy production rate (via instantaneous)")

plt.xlabel('delta')
plt.ylabel('epr')

plt.savefig("42.OLD.EPR.via_inst.png")


# Plotting entropy
plt.figure(43)
plt.clf()
OU.plot_cool_colourline2(deltas, epr_v, deltas, lw=1)

plt.suptitle("Entropy")

plt.xlabel('time')
plt.ylabel('H')
for i in range(n):
    plt.plot(H[:, i])

# plotting instantanous epr
plt.figure(44)
plt.clf()
plt.suptitle("Instantaneous epr")

plt.xlabel('time')
plt.ylabel('Inst epr')
for i in range(n):
    plt.plot(epr_inst[epr_inst[:, i] >= 0, i])
