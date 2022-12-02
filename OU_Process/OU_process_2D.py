'''
IMPORTS
'''

import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

'''
OU Process
'''


class OU_process(object):

    def __init__(self, dim, friction, volatility):
        super(OU_process, self).__init__()  # constructs the object instance
        if d != 2:
            raise TypeError("This code is exclusively for 2D!")
        self.d = dim
        self.B = friction
        self.sigma = volatility

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
            x[:, t, :] = x[:, t - 1, :] - epsilon * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                         + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)
            if np.count_nonzero(np.isnan(x)):
                raise TypeError("nan")
        return x

    def showpaths(self, x, color='blue', label='helloword'):  # compute and plot statistics of the simulation
        [d, T, N] = np.shape(x)
        '''
        Sample trajectory
        '''

        plt.figure(1)
        plt.suptitle("OU process trajectory")
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plot_colourline(x[0, :, 0], x[1, :, 0], c=x[0, :, 0], lw=0.1)
        #plt.legend()
        plt.savefig("OUprocess.hypo.2D.png")

    def showentropy(self, x, nbins=10, color='blue',
                    label='helloword'):  # compute and plot statistics of the simulation
        [d, T, N] = np.shape(x)
        '''
        Entropy
        '''

        plt.figure(2)
        plt.suptitle("Entropy OU process")

        entropy_x = entropy(x, nbins)

        plt.xlabel('time')
        plt.ylabel('H[p(x)]')
        plt.plot(range(T), entropy_x, c=color, linewidth=0.5, label=label)
        plt.legend()
        plt.savefig("OUprocess.entropy.2D.png")

    def attributes(self):
        return [self.B, self.sigma]


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


def bin_range(x):
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]), np.max(x[d, :, :])])
    return b_range


def ent_prod_rate_2D(process):
    [B, sigma] = process.attributes()
    D = sigma @ sigma.T / 2  # diffusion matrix
    a = B[0, 0]
    b = B[0, 1]
    c = B[1, 0]
    d = B[1, 1]
    u = D[0, 0]
    v = D[1, 1]
    w = D[1, 0]
    q = c * u - b * v + (d - a) * w  # irreversibility parameter
    phi = q ** 2 / ((a + d) * np.linalg.det(D))
    return phi

def plot_colourline(x,y,c, lw =0.5):
    c = cm.cool((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i], linewidth=lw)
    return


'''
Clear figures
'''
for i in np.arange(1,4):
    plt.figure(i)
    plt.clf()

'''
MAIN
'''

seed = 1
np.random.seed(seed)

d = 2  # dimension of state-space
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

B1 = np.array([0, 1, -1, 0]).reshape([2, 2]) #np.eye(d)  # symmetric matrix
B2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
sigma = np.array([1, 0, 1, 0]).reshape([2, 2])  #np.eye(d)  # volatility matrix

'''
Run paths
'''
n = 2  # number of simulations
deltas = [0] #np.linspace(0, 1 - 1e-10, n)
T = 10 ** 4  # number of time-steps
N = 2  # number of trajectories
i = 0  # set counter
for delta in deltas:
    delta = np.round(delta,1)
    process = OU_process(dim=d, friction=B1 + delta * B2, volatility=sigma)
    x = process.simulation(x0, epsilon, T, N)
    process.showpaths(x, color=cm.cool(delta), label=f"delta = {delta}")
    i += 1

'''
Run entropy
'''
T = 100  # number of time-steps
N = 10 ** 4  # number of trajectories
i = 0  # set counter
for delta in deltas:
    delta = np.round(delta, 1)
    process = OU_process(dim=d, friction=B1 + delta * B2, volatility=sigma)
    x = process.simulation(x0, epsilon, T, N)

    plt.figure(5)
    plt.clf()
    plt.scatter(x[0,-1,:], x[1,-1,:])

    plt.figure(6)
    plt.clf()
    plt.suptitle('Distribution after some time')
    plt.hist2d(x[0,-1,:], x[1,-1,:], bins=(50, 50), cmap=cm.jet)
    plt.savefig("OUprocess.hypo.law.png")
    process.showentropy(x, color=cm.cool(delta), label=f"delta = {delta}")
    i += 1

'''
Compute Entropy production rate
'''
n = 100  # number of calculations
epr = np.zeros([n])  # epr
finedeltas = np.linspace(0, 1 - 1e-10, n)
i = 0  # set counter
for delta in finedeltas:
    process = OU_process(dim=d, friction=B1 + delta * B2, volatility=sigma)
    epr[i] = ent_prod_rate_2D(process)
    i += 1

'''
Plot coloured entropy production rate
'''

fig = plt.figure(3)
plt.suptitle("Entropy production rate")

plt.xlabel('delta')
plt.ylabel('epr')
#ax  = fig.add_subplot(111)
plot_colourline(finedeltas,epr,finedeltas, lw=1)
i=0
for delta in deltas:
    index = np.argmin(np.abs(finedeltas - delta))
    label = np.round(finedeltas[index],1)
    #plt.scatter(finedeltas[index], epr[index], color = cm.cool(finedeltas[index]), label= f'delta = {label}', s=10)

#plt.legend()
#plt.savefig("OUprocess.epr.2D.png")


'''
Print entropy production rate
'''
print(deltas)
print(epr)