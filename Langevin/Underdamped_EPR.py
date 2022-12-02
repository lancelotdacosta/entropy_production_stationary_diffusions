'''
Underdamped Langevin dynamics in a quadratic potential
'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import OU_process_empirical_EPR as OU

'''
Class
'''


class underdamped_process(object):

    def __init__(self, dim, gradV, sigma, friction=0.01):
        super(underdamped_process, self).__init__()  # constructs the object instance
        if dim != 2:
            raise TypeError("This code is exclusively for 2D!")
        self.d = dim
        self.gradV = gradV
        self.sigma = sigma  # volatility matrix
        if gamma <= 0:
            raise TypeError('gamma non-positive')
        self.gamma = friction  # friction

    def simulation(self, x0, epsilon=0.01, T=100, N=1):  # run OU process for multiple trajectories
        w = np.random.normal(0, np.sqrt(epsilon), T * self.d * N).reshape([self.d, T, N])  # random Wiener fluctuations
        w = np.tensordot(self.sigma, w[:, :, :], axes=1) #multiply with constant volatility
        x = np.zeros([self.d, T, N])  # store values of the process
        v = np.zeros([self.d, N])  # initialise process velocity to zero
        '''
        check initial condition
        '''
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        '''
        perform simulation
        '''
        for t in range(1, T):
            v = (1 - epsilon) * v \
                - epsilon / self.gamma * np.tensordot(self.gradV, x[:, t - 1, :], axes=1) + w[:, t - 1, :] #update velocity
            x[:, t, :] = x[:, t - 1, :] + epsilon / self.gamma * v  # update position
            if np.count_nonzero(np.isnan(x)):
                raise TypeError("nan")
        return x

    def attributes(self):
        return [self.gradV, self.sigma, self.gamma]


'''
MAIN
'''

seed = 1
np.random.seed(seed)


'''
Figure 50. EPR via instantaneous epr
'''

print("Starting Figure 50")

d = 2  # dimension
n = 1  # number of simulations
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step
deltas = np.linspace(0, 1 - 0.01, n)

gradV = np.eye(d)  # symmetric matrix
sigma = np.eye(d)  # volatility matrix
gamma =10**(-2) #friction coefficient

N = 10 ** 5  # number of trajectories (used to estimate the law of the process via histogram method)
T = 1 * 10 ** 4  # number of timesteps to run the process over

epr_v_ULD = np.empty([n])  # epr via instantaneous
epr_v2_ULD = np.empty([n])  # epr via instantaneous median
H = -np.ones([T, n])  # score entropy
epr_inst = -np.ones([T, n])  # instantaneous entropy production rate
# pos = np.zeros([d, T, 1, n])  # positions

i = 0  # set counter
for delta in deltas:
    print(np.round(i / n * 100, 2))
    process = underdamped_process(dim=d, gradV=gradV, volatility=sigma, friction =gamma)
    x= np.zeros([d,2,N])
    t = 0
    while t < T:
        steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        # pos[:, t:(t + steps), 0, i] = x[:, :, 0]  # record positions
        epr_inst[t:(t + steps - 1), i] = OU.inst_epr(x, epsilon, nbins=10)  # record instantaneous entropy production
        H[t:(t + steps), i] = OU.entropy(x, nbins=10)  # record entropy
        t += steps
    epr_v_ULD[i] = np.mean(epr_inst[epr_inst[:, i] >= 0, i])
    epr_v2_ULD[i] = np.median(epr_inst[epr_inst[:, i] >= 0, i])
    i += 1

# Plotting epr
plt.figure(50)
plt.clf()
OU.plot_cool_colourline2(deltas, epr_v_ULD, deltas, lw=1)

plt.suptitle("Entropy production rate (via instantaneous)")

plt.savefig("50.ULD.EPR.via_inst.png")


# Plotting entropy
plt.figure(53)
plt.clf()

plt.suptitle("Entropy")

plt.xlabel('time')
plt.ylabel('H')
for i in range(n):
    plt.plot(H[:, i])

# plotting instantanous epr
plt.figure(54)
plt.clf()
plt.suptitle("Instantaneous epr")

plt.xlabel('time')
plt.ylabel('Inst epr')
for i in range(n):
    plt.plot(epr_inst[epr_inst[:, i] >= 0, i])
