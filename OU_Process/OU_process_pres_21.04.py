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
        n = np.shape(self.B)[0]
        S= np.eye(n) #stationary covariance
        for i in range(n):
            for j in range(n):
                S[i,j]= scipy.integrate.quad(func=integrand, a=0, b=np.inf, args=(self.B,self.sigma, i,j))[0]
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
Figure 1: Brownian path 1d time
'''
print('Starting fig 1')
dim = 2  # dimension of state-space
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

B = np.zeros(dim)  # drift
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

n = 2  # number of simulations
deltas = [1]  # np.linspace(0, 1 - 1e-10, n)
T = 10 ** 3  # number of time-steps
N = 3  # number of trajectories
i = 0  # set counter

process = OU_process(dim=dim, friction=B, volatility=sigma)
x = process.simulation(x0, epsilon, T, N)

plt.figure(1)
plt.plot(range(1, T + 1), x[0, :, 0], label=f"BM 1D", linewidth=0.3)
plt.plot(range(1, T + 1), x[1, :, 0], label=f"BM 1D", linewidth=0.3)
# plt.plot(range(1,T+1), x[0,:,1], label = f"BM 1D", linewidth =0.3)
plt.plot(range(1, T + 1), x[1, :, 2], label=f"BM 1D", linewidth=0.3)
plt.suptitle('1D Brownian motion sample paths')
plt.xlabel('time')
plt.ylabel('BM')

plt.savefig("1.BM.1D.vs.time.png")
plt.clf()

'''
Figure 2: BM 2D sample path colourjet
'''
print('Starting fig 2')

dim = 2  # dimension of state-space
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

B = np.zeros(dim)  # drift
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

T = 10 ** 4  # number of time-steps
N = 2  # number of trajectories
i = 0  # set counter

process = OU_process(dim=dim, friction=B, volatility=sigma)
x = process.simulation(x0, epsilon, T, N)

plt.figure(2)
plot_cool_colourline(x[1, :, 1], x[0, :, 1], c=np.arange(T), lw=0.3)
plt.suptitle('2D Brownian motion sample paths')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("2.BM.2D.paths.colourjet.png")
plt.clf()

'''
Figure 3: Example of a confining potential in 1D and OU process dynamics. B =Id, sigma =Id
    Insert parabola, and scatter dynamics on it colourjet
'''
print('Starting fig 3')

dim = 2  # dimension of state-space
x0 = np.array([2, 2])  # initial condition
epsilon = 0.1  # time-step

B = np.eye(dim)  # drift
sigma = 0.1*np.eye(dim)  # np.eye(d)  # volatility matrix

T = 10 ** 2  # number of time-steps
N = 2  # number of trajectories

process = OU_process(dim=dim, friction=B, volatility=sigma)
x = process.simulation(x0, epsilon, T, N)

parabola_domain = np.linspace(np.min(x[0, :, 0])-0.5, np.max(x[0, :, 0])+0.5, 10 ** 5)
plt.figure(3)
plt.clf()
plt.plot(parabola_domain, 0.5 * parabola_domain ** 2, c='black',zorder=-1)
plot_cool_scatter(x[0, :, 0], 0.5 * x[0, :, 0] ** 2, c=np.arange(T), lw=10)
plt.xlabel('X_t')
plt.ylabel('V(X_t)')
plt.savefig("3.OLD.1D.confining potential.png")
plt.clf()
del parabola_domain

'''
Figure 4: same dynamics, plotted against time
'''
print('Starting fig 4')

dim = 2  # dimension of state-space
x0 = np.array([2, 2])  # initial condition

B = np.eye(dim)  # drift
sigma = 0.1*np.eye(dim)  # np.eye(d)  # volatility matrix
epsilon = 0.01  # time-step

B = np.eye(dim)  # drift
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

T = 10 ** 3  # number of time-steps
N = 2  # number of trajectories

process = OU_process(dim=dim, friction=B, volatility=sigma)
x = process.simulation(x0, epsilon, T, N)

plt.figure(4)
plt.clf()
plot_cool_colourline(range(T), x[0, :, 0],range(T))
plt.xlabel('time')
plt.ylabel('X_t')
plt.savefig("4.OLD.1D.confining potential.time.overdamped.png")
plt.clf()

'''
Figure 5: OU process path: 1 dimensional, B= sigma = 1. Graphic 1D vs time
'''
print('Starting fig 5')
dim = 2  # dimension of state-space
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

B = np.eye(dim)  # drift
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

n = 2  # number of simulations
T = 10 ** 3  # number of time-steps
N = 2  # number of trajectories

process = OU_process(dim=dim, friction=B, volatility=sigma)
x = process.simulation(x0, epsilon, T, N)


plt.figure(5)
plt.clf()
plot_cool_colourline(range(T), x[0, :T, 0],range(T), lw=0.3)
# plt.plot(range(1, Tprime + 1), x[1, :Tprime, 0], linewidth=0.3)
# plt.plot(range(1,T+1), x[0,:,1], label = f"BM 1D", linewidth =0.3)
# plt.plot(range(1, Tprime + 1), x[1, :Tprime, 1], linewidth=0.3)
plt.suptitle('1D OU process sample paths')
plt.xlabel('time')
plt.ylabel('OU process')

plt.savefig("5.OU.1D.vs.time.png")
plt.clf()

'''
Figure 6: OU process path:  2 dimensional, B= sigma = Id: Insert graphic 2D colour jet
'''
T = 10**3

print('Starting fig 6')
plt.figure(6)
plot_cool_colourline(x[1, :T, 1], x[0, :T, 1], c=np.arange(T), lw=0.3)
plt.suptitle('2D OU process sample paths')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("6.OUprocess.2D.paths.colourjet.png")
plt.clf()

'''
Figure 7.8.9:
3 simulations OU process in 2D, one with negative eigenvalues of B, one with positive and one with null

Figure 10.11.12: Heat map of corresponding to laws of fig 6.7.8
'''
print('Starting fig 7-12')
dim = 2  # dimension of state-space
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

B_neg = -np.eye(dim)  # drift
B_null = np.zeros(dim)  # drift
B_pos = np.eye(dim)  # drift
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

n = 2  # number of simulations
T = 5*10 ** 3  # number of time-steps
N = 10 ** 3  # number of trajectories

process_neg = OU_process(dim=dim, friction=B_neg, volatility=sigma)
x = process_neg.simulation(x0, epsilon, T, N)

plt.figure(7)
plot_cool_colourline(x[0, :, 1], x[1, :, 1], c=np.arange(T), lw=0.3)
plt.suptitle('2D OU process paths eigenvalues <0')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("7.OUprocess.2D.paths.eigen<0.png")
plt.clf()

plt.figure(10)
plt.hist2d(x[0, -1, :], x[1, -1, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('2D OU process law eigenvalues <0')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("10.OUprocess.2D.law.eigen<0.png")
plt.clf()

#print('7,10 done')
process_null = OU_process(dim=dim, friction=B_null, volatility=sigma)
x = process_null.simulation(x0, epsilon, T, N)

plt.figure(8)
plot_cool_colourline(x[0, :, 1], x[1, :, 1], c=np.arange(T), lw=0.3)
plt.suptitle('2D OU process paths eigenvalues =0')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("8.OUprocess.2D.paths.eigen=0.png")
plt.clf()

plt.figure(11)
plt.hist2d(x[0, -1, :], x[1, -1, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('2D OU process law eigenvalues =0')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("11.OUprocess.2D.law.eigen=0.png")
plt.clf()

process_pos = OU_process(dim=dim, friction=B_pos, volatility=sigma)
x = process_pos.simulation(x0, epsilon, T, N)

plt.figure(9)
plot_cool_colourline(x[0, :, 1], x[1, :, 1], c=np.arange(T), lw=0.3)
plt.suptitle('2D OU process paths eigenvalues >0')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("9.OUprocess.2D.paths.eigen>0.png")
plt.clf()

plt.figure(12)
plt.hist2d(x[0, -1, :], x[1, -1, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('2D OU process law eigenvalues >0')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("12.OUprocess.2D.law.eigen>0.png")
plt.clf()

clear_figures(12)
'''
        For 3 values of delta =0,0.5,1:
Figure 13.14.15		paths simulations of OU at ESS/NESS  with colourjet
Figure 16.17.18 	simulations of ESS/NESS law heat map
Figure 19.20.21 	simulations of vector field stationary probability current
'''
print('Starting fig 13-21')
dim = 2  # dimension of state-space
# x0 = np.array([0, 0])  # initial condition
epsilon = 0.001  # time-step

T = 10 ** 4  # number of time-steps
N = 2  # number of trajectories
heatmap_no = 10 ** 4  # number of samples for heat map

B1 = np.eye(dim)  # symmetric matrix
B2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
sigma = np.eye(dim)  # np.eye(d)  # volatility matrix

i = 0  # set counter
deltas = np.linspace(0, 1 - 1e-5, 3)
for d in deltas:
    process = OU_process(dim=dim, friction=B1 + d * B2, volatility=sigma)
    delta = np.round(d, 1)
    S = process.stat_cov2D() #They all have the same covariance!
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

    x0 = x0[:, :N]
    x = process.simulation(x0, epsilon, T, N)

    figno = 13 + i
    plt.figure(figno)
    plot_cool_colourline2(x[0, :, 1], x[1, :, 1], c=np.arange(T), lw=0.2)
    plt.suptitle(f'2D OU process NESS paths delta={delta}')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig(f"{figno}.OUprocess.NESS.2D.paths.delta={delta}.png")
    plt.clf()
    print(f'{figno} done')
    del x

    # plot probability current
    figno = 19 + i
    plt.figure(figno)
    plt.clf()
    [B, sigma] = process.attributes()
    D = sigma @ sigma.T / 2
    Jmx = B - D @ np.linalg.inv(S)  # matrix that is part of the probability current
    width = 10
    X= np.arange(-width, width, 1)
    Y= np.arange(-width, width, 1)
    for k in Y:
        for j in X:
            [u, v] = -Jmx @ np.array([j, k]) * multivariate_normal.pdf(np.array([j, k]), mean=np.array([0, 0]), cov=S)
            print([u,v])
            plt.quiver(j, k, u, v)

    plt.suptitle(f'2D OU process NESS current delta={delta}')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.savefig(f"{figno}.OUprocess.NESS.2D.Jcurrent.delta={delta}.png")
    #plt.clf()
    print(f'{figno} done')

    i += 1

'''
Figure 22. figure entropy production rate as a function of irreversibility
'''
d=2
n = 100  # number of simulations
x0 = np.array([0, 0])  # initial condition
epsilon = 0.01  # time-step

B1 = np.eye(d)  # symmetric matrix
B2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
sigma = np.array([1, 0, 1, 0]).reshape([2, 2])  #np.eye(d)  # volatility matrix

epr = np.zeros([n])  # epr
deltas = np.linspace(0, 1 - 1e-10, n)
i = 0  # set counter
for delta in deltas:
    process = OU_process(dim=d, friction=B1 + delta * B2, volatility=sigma)
    epr[i] = ent_prod_rate_2D(process)
    i += 1

plt.figure(22)
plot_cool_colourline2(deltas,epr,deltas, lw=1)

plt.suptitle("Entropy production rate")

plt.xlabel('delta')
plt.ylabel('epr')
plt.savefig("22.OUprocess.epr(delta).png")
plt.clf()


'''
Figure 23. 24. Trajectories not hypoelliptic VS hypoelliptic 
Figure 25. 26. Heat map law after some time not hypoelliptic VS hypoelliptic
'''
d=2
B_not_hypo = np.eye(2) # drift
B_hypo = np.array([0, 1, -1, 0]).reshape([2, 2])
sigma = np.array([1,0,1,0]).reshape([d,d])

n = 2  # number of simulations
T = 10 ** 4  # number of time-steps
N = 10 ** 3  # number of trajectories

process_not_hypo = OU_process(dim=d, friction=B_not_hypo, volatility=sigma)
x = process_neg.simulation(x0, epsilon, T, N)

plt.figure(23)
plot_cool_colourline2(x[0, :, 0], x[1, :, 0], c=np.arange(T), lw=0.3)
plt.suptitle('2D OU process paths not hypoelliptic')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("23.OUprocess.degen.nothypo.paths.png")
plt.clf()

plt.figure(25)
plt.hist2d(x[0, -1, :], x[1, -1, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('2D OU process longtime distribution not hypoelliptic')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("25.OUprocess.degen.nothypo.longtimelaw.png")
plt.clf()


process_hypo = OU_process(dim=d, friction=B_hypo, volatility=sigma)
x = process_neg.simulation(x0, epsilon, T, N)

plt.figure(24)
plot_cool_colourline2(x[0, :, 0], x[1, :, 0], c=np.arange(T), lw=0.3)
plt.suptitle('2D OU process paths hypoelliptic')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("24.OUprocess.degen.hypo.paths.png")
plt.clf()

plt.figure(26)
plt.hist2d(x[0, -1, :], x[1, -1, :], bins=(50, 50), cmap=cm.jet)
plt.suptitle('2D OU process longtime distribution hypoelliptic')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.savefig("26.OUprocess.degen.hypo.longtimelaw.png")
plt.clf()

