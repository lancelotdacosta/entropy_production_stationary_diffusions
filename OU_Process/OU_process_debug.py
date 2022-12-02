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

'''
Process
'''


class OU_process(object):

    def __init__(self, dim, friction, volatility):
        super(OU_process, self).__init__()  # constructs the object instance
        if dim != 2:
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
        plt.plot(x[0, :, 0], x[1, :, 0], c=color, linewidth=0.1, label=label)
        plt.legend()
        plt.savefig("OUprocess.trajectory.2D.png")

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

    def stationary_covariance(self):
        '''
        D = self.sigma @ self.sigma.T / 2  # diffusion matrix
        a = self.B[0, 0]
        b = self.B[0, 1]
        c = self.B[1, 0]
        d = self.B[1, 1]
        u = D[0, 0]
        v = D[1, 1]
        w = D[1, 0]
        detB = np.linalg.det(self.B)
        S = np.array([(detB + d ** 2) * u + b ** 2 * v - 2 * b * d * w,
                      -c * d * u - a * b * v + 2 * a * d * w,
                      -c * d * u - a * b * v + 2 * a * d * w,
                      c ** 2 * u + (detB + a ** 2) * v - 2 * a * c * w]).reshape(2,2)
        return S / ((a + d) * detB)
        '''
        n = np.shape(B)[0]
        S= np.eye(n) #stationary covariance
        for i in range(n):
            for j in range(n):
                S[i,j]= scipy.integrate.quad(func=integrand, a=0, b=np.inf, args=(B1,sigma, i,j))[0]
        return S

    def stat_cov2D(self):
        D = self.sigma @ self.sigma.T / 2  # diffusion matrix
        a = self.B[0, 0]
        b = self.B[0, 1]
        c = self.B[1, 0]
        d = self.B[1, 1]
        u = D[0, 0]
        v = D[1, 1]
        w = D[1, 0]
        detB = np.linalg.det(self.B)
        S = np.array([(detB + d ** 2) * u + b ** 2 * v - 2 * b * d * w,
                      -c * d * u - a * b * v + 2 * a * d * w,
                      -c * d * u - a * b * v + 2 * a * d * w,
                      c ** 2 * u + (detB + a ** 2) * v - 2 * a * c * w]).reshape(2,2)
        return S / ((a + d) * detB)


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

def integrand(t, B, sigma, i, j): #to compute stationary covariance
    mx = scipy.linalg.expm(-B * t) @ sigma
    mx = mx @ mx.T
    return mx[i,j]

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


# cool colourline
def plot_cool_colourline2(x, y, c, lw=0.5):
    c = cm.cool((c - np.min(c)) / (np.max(c) - np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
    return

def plot_cool_colourline(x, y, c, lw=0.5):
    c = cm.cool((c - np.min(c)) / (np.max(c) - np.min(c)))
    plt.plot(x, y, c=c, linewidth=lw)
    plt.show()
    return


def plot_cool_scatter(x, y, c, lw=0.5):
    c = cm.cool((c - np.min(c)) / (np.max(c) - np.min(c)))
    plt.scatter(x, y, c=c, s=lw, zorder=1)
    plt.show()
    return


# magma colourline
def plot_magma_colourline(x, y, c, lw=0.5):
    c = cm.magma((c - np.min(c)) / (np.max(c) - np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
    return

def clear_figures(f):
    for i in np.arange(1, f+1):
        plt.figure(i)
        plt.clf()


'''
MAIN
'''

seed = 1
np.random.seed(seed)


'''
Figures
'''
print('Starting fig 13-21')
dim = 2  # dimension of state-space
# x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

T = 10 ** 4  # number of time-steps
N = 2  # number of trajectories
heatmap_no = 10 ** 4  # number of samples for heat map

B1 = np.eye(dim)  # symmetric matrix
B2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

i = 0  # set counter
deltas = [0] #np.linspace(0, 1 - 1e-5, 3)
for d in deltas:
    process = OU_process(dim=dim, friction=B1 + d * B2, volatility=sigma)
    delta = np.round(d, 1)
    S = process.stat_cov2D()
    x0 = np.random.multivariate_normal(mean=np.array([0, 0]), cov=S, size=heatmap_no).T
    # sample from stationary covariance

    figno = 16 + i
    plt.figure(figno)
    plt.hist2d(x0[0, :], x0[1, :], bins=(50, 50), cmap=cm.jet)
    plt.suptitle(f'2D OU process NESS heatmap delta={delta}')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig(f"{figno}.OUprocess.NESS.2D.heatmap.delta={delta}.png")
    plt.clf()
    print(f'{figno} done')

    x0 = x0[:, 0:N]
    x = process.simulation(x0, epsilon, T, N)

    figno = 13 + i
    plt.figure(figno)
    plot_cool_colourline(x[0, :, 0], x[1, :, 0], c=np.arange(T), lw=0.3)
    plt.suptitle(f'2D OU process NESS paths delta={delta}')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig(f"{figno}.OUprocess.NESS.2D.paths.delta={delta}.png")
    plt.clf()
    print(f'{figno} done')

    # plot probability current
    figno = 19 + i
    plt.figure(figno)

    [B, sigma] = process.attributes()
    D = sigma @ sigma.T / 2
    Jmx = B + D @ np.linalg.inv(S)  # matrix that is part of the probability current
    width = 10
    X, Y = np.arange(-width, width, 1)
    for k in Y:
        for j in X:
            [u, v] = - Jmx @ np.array([j, k]) * multivariate_normal.pdf([j, k], mean=np.array([0, 0]), cov=S)
            plt.quiver(j, k, u, v)


    #plt.quiver(x[0, :, 1], x[1, :, 1], c=np.arange(T), lw=0.3)
    plt.suptitle(f'2D OU process NESS current delta={delta}')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig(f"{figno}.OUprocess.NESS.2D.Jcurrent.delta={delta}.png")
    plt.clf()
    print(f'{figno} done')

    i += 1
