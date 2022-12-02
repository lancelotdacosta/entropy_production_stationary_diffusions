'''
IMPORTS
'''

import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib.pyplot as plt

'''
OU Process
'''


class OU_process(object):

    def __init__(self, dim, drift, diffusion):
        super(OU_process, self).__init__()  # constructs the object instance
        self.d = dim
        self.A = drift
        self.B = diffusion

    def simulation(self, x0, epsilon=0.01, T=100, N=1):  # run OU process for multiple trajectories
        w = np.random.normal(0, np.sqrt(epsilon), T * self.d * N).reshape([self.d, T, N])  # random fluctuations
        x = np.zeros([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        for t in range(1, T):
            x[:, t, :] = x[:, t - 1, :] - epsilon * np.tensordot(self.A, x[:, t - 1, :], axes=1) \
                         + np.tensordot(self.B, w[:, t - 1, :], axes=1)
            if np.count_nonzero(np.isnan(x)):
                raise TypeError("nan")
        return x

    def show(self, x, epsilon, nbins=10, color='blue'):  # compute and plot statistics of the simulation
        [d, T, N] = np.shape(x)
        '''
        Sample trajectory
        '''

        plt.figure(1)
        plt.suptitle("OU process trajectory")
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.plot(x[0, :, 0], x[1, :, 0], c=color, linewidth=0.1)
        plt.savefig("OUprocess.trajectory.png")

        '''
        Entropy
        '''

        plt.figure(2)
        plt.suptitle("Entropy OU process")

        entropy_x = entropy(x, nbins)

        plt.xlabel('time')
        plt.ylabel('H[p(x)]')
        plt.plot(range(T), entropy_x, c=color, linewidth=0.1)
        plt.savefig("OUprocess.entropy.png")

        '''
        (Infinitessimal) Entropy production rate
        '''
        plt.figure(3)
        plt.suptitle("Infinitessimal entropy production rate")

        inf_epr = inf_ent_prod_rate(x, epsilon, nbins)

        plt.xlabel('time')
        # plt.ylabel('H[p(x)]')
        # plt.yscale('log')
        plt.plot(range(1, T - 1), inf_epr[1:], c=color, linewidth=0.1)
        plt.savefig("OUprocess.inf_epr.png")

        '''
        Convolved Entropy production rate
        '''
        plt.figure(4)
        plt.suptitle("Entropy production rate (convolved)")

        sigma = 1
        dx = (3940 - 3930) / N
        gx = np.arange(-3 * sigma, 3 * sigma, dx)
        gaussian = np.exp(-(gx / sigma) ** 2 / 2)
        ent_prod_x_conv = np.convolve(inf_epr, gaussian, mode="valid")

        plt.xlabel('time')
        plt.yscale('log')
        plt.plot(range(len(ent_prod_x_conv)), ent_prod_x_conv + 1, c=color, linewidth=0.1)
        plt.savefig("OUprocess.inf_epr.conv.png")

        return np.median(inf_epr)

    def attributes(self):
        return [self.A, self.B]


'''
Functions
'''


def entropy(x, nbins=10):
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)
    entropy = np.zeros(T)
    for t in range(T):
        h = np.histogramdd(x[:, t, :].T, bins=nbins, range=b_range)[0]
        entropy[t] = - np.sum(h / N * np.log(h + (h == 0)))

    return entropy + np.log(N)


def inf_ent_prod_rate(x, epsilon, nbins=10):
    [d, T, N] = np.shape(x)
    b_range = bin_range(x) * 2

    inf_epr = np.zeros([T - 1])

    for t in range(T - 1):
        x_t = x[:, t:t + 2, :]  # trajectories at time t, t+1
        x_mt = np.flip(x_t, axis=1)  # trajectories at time t+1, t (time reversal)
        x_t = x_t.reshape([d * 2, N])  # samples from (x_t, x_t+1)
        x_mt = x_mt.reshape([d * 2, N])  # samples from (x_t+1, x_t) (time reversal)
        e = np.histogramdd(x_t.T, bins=nbins, range=b_range)[0]  # law of (x_t, x_t+1) (unnormalised)
        h = np.histogramdd(x_mt.T, bins=nbins, range=b_range)[0]  # law of (x_t+1, x_t) (unnormalised)
        nonzero = (e != 0) * (h != 0)  # shows where e and h are non-zero
        zero = (nonzero == 0)  # shows where e or h are zero
        inf_epr[t] = np.sum(
            e / (N * epsilon) * np.log((e * nonzero + zero) / (h * nonzero + zero)))  # 1/epsilon * KL divergence

    return inf_epr  # 1/epsilon * KL divergence


def ent_prod_rate(x, t=4, nbins=5):
    [d, T, N] = np.shape(x)

    b_range = bin_range(x) * t
    x = x[:, -t:, :]  # take only the last times
    x_r = np.flip(x, axis=1)  # trajectories at time t+1, t (time reversal)
    x = x.reshape([d * t, N])  # samples from (x_t, x_t+1)
    x_r = x_r.reshape([d * t, N])  # samples from (x_t+1, x_t) (time reversal)
    e = np.histogramdd(x.T, bins=nbins, range=b_range)[0]  # law of (x_t, x_t+1) (unnormalised)
    h = np.histogramdd(x_r.T, bins=nbins, range=b_range)[0]  # law of (x_t+1, x_t) (unnormalised)
    nonzero = (e != 0) * (h != 0)  # shows where e and h are non-zero
    zero = (nonzero == 0)  # shows where e or h are zero
    epr = np.sum((e / (N * T)) * np.log((e * nonzero + zero) / (h * nonzero + zero)))  # deal with negative values here

    return epr


def ent_prod_rate_paper(process):
    [B, D] = process.attributes()
    # D = D/2
    S = 0
    for t in np.linspace(10 ** (-6), 10 ** 3, 10 ** 6):
        ebt = expm(-B * t)
        S += 0.001 * ebt * D * ebt.T
    S = S * 2
    Q = B * S - D
    return -np.trace(np.linalg.inv(D) * B * Q)


def ent_prod_rate_paper2(process):
    [B, D] = process.attributes()
    D = D / 2
    S = 0
    for t in np.linspace(10 ** (-6), 10 ** 3, 10 ** 6):
        ebt = expm(-B * t)
        S += 0.001 * ebt * D * ebt.T
    S = S * 2
    Q = B * S - D
    return -np.trace(np.linalg.inv(D) * B * Q)


def bin_range(x):
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]), np.max(x[d, :, :])])
    return b_range


'''
MAIN
'''

seed = 1
np.random.seed(seed)

d = 2  # dimension of state-space
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step
T = 4  # number of time-steps
N = 10 ** 6  # number of trajectories
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
A1 = np.eye(d)  # symmetric matrix
A2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
B = np.eye(d)  # diffusion matrix
n = 31  # number of simulations
epr = np.zeros([n, 4])  # epr estimators
i = 0  # set counter

for delta in np.linspace(0, 30, n):
    process = OU_process(dim=d, drift=A1 + delta * A2, diffusion=B)
    for t in range(10 ** 1):
        x = process.simulation(x0, epsilon, T, N)
        x0 = x[:, -1, :]
    epr[i, 3] = process.show(x, epsilon, nbins=10, color=colors[i % 7])  # estimated via infinitessimal epr
    epr[i, 2] = ent_prod_rate(x, t=4)  # empirical epr
    epr[i, 1] = ent_prod_rate_paper2(process)  # estimator from paper bis
    epr[i, 0] = ent_prod_rate_paper(process)  # estimator from paper
    i += 1

plt.figure(5)
plt.suptitle("Entropy production rate")

plt.xlabel('time')
plt.plot(range(n), epr[:, 0], linewidth=0.5)  # estimator paper
plt.plot(range(n), epr[:, 1], linewidth=0.2, c='black')  # estimator paper 2
plt.plot(range(n), epr[:, 2], linewidth=0.5, c='green')  # empirical
# plt.plot(range(n), epr[:, 3], linewidth=0.5, c='yellow')  # estimated via infinitessimal epr

print(epr[:, 0])
print(epr[:, 1])
print(epr[:, 2])
print(epr[:, 3])

plt.savefig("OUprocess.epr.png")
