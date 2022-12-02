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
import OU_process_functions as ou

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
        S = np.eye(n)  # stationary covariance
        for i in range(n):
            for j in range(n):
                S[i, j] = scipy.integrate.quad(func=integrand, a=0, b=np.inf, args=(self.B, self.sigma, i, j))[0]
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
                      c ** 2 * u + (detB + a ** 2) * v - 2 * a * c * w]).reshape(2, 2)
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


def inst_epr(x, epsilon, nbins=10):
    [d, T, N] = np.shape(x)
    b_range = bin_range(x) * 2  # double the list

    inf_epr = np.zeros([T - 1])

    for t in range(T - 1):
        x_t = x[:, t:t + 2, :]  # trajectories at time t, t+1
        x_mt = np.flip(x_t, axis=1)  # trajectories at time t+1, t (time reversal)
        x_t = x_t.reshape([d * 2, N])  # samples from (x_t, x_t+1)
        x_mt = x_mt.reshape([d * 2, N])  # samples from (x_t+1, x_t) (time reversal)
        e = np.histogramdd(x_t.T, bins=nbins, range=b_range)[0]  # law of (x_t, x_t+1) (unnormalised)
        h = np.histogramdd(x_mt.T, bins=nbins, range=b_range)[0]  # law of (x_t+1, x_t) (unnormalised)
        #nonzero = (e != 0)*(h != 0)
        #inf_epr[t]= np.sum(np.where(nonzero, e/N * np.log(e / h), 0)) / epsilon
        nonzero = (e != 0) * (h != 0)  # shows where e and h are non-zero
        zero = (nonzero == 0)  # shows where e or h are zero
        inf_epr[t] = np.sum(
            e / (N * epsilon) * np.log((e * nonzero + zero) / (h * nonzero + zero)))  # 1/epsilon * KL divergence
    return inf_epr  # 1/epsilon * KL divergence


def stationary_density(x, nbins=10):  # return unnormalised stationary density
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)

    x = x.reshape([d, N * T])  # samples from (x_t, x_t+1)

    hist = np.histogramdd(x.T, bins=nbins, range=b_range)[0]  # stationary law of x unnormalised

    # hist= hist/np.sum(hist) # stationary law of x normalised to probabilities

    mu = hist.reshape([nbins ** d])
    return mu


def stationary_trans_density(x, nbins=10):  # return normalised transition matrix and unnormalised stationary density
    [d, T, N] = np.shape(x)
    b_range = bin_range(x) * 2  # double the list

    x = np.append(x[:, :-1, :], x[:, 1:, :], axis=0)

    P = np.zeros([nbins ** d, nbins ** d])
    for t in range(T - 1):
        x_tt = x[:, t, :].reshape([2 * d, N])  # x_t,x_t+1
        hist = np.histogramdd(x_tt.T, bins=nbins, range=b_range)[0]  # stationary law of x unnormalised
        h = hist.reshape([nbins ** d, nbins ** d])
        P += h.T  # transpose so that it corresponds to an unnormalised stochastic matrix

    mu = np.sum(P, axis=0)  # unnormalised stationary distribution
    mu = mu / np.sum(mu)  # stationary density
    P = P / np.sum(P, axis=0)  # transition probabilities (stochastic matrix)
    return P, mu


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


def integrand(t, B, sigma, i, j):  # to compute stationary covariance
    mx = scipy.linalg.expm(-B * t) @ sigma
    mx = mx @ mx.T
    return mx[i, j]


def bin_range(x):
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]), np.max(x[d, :, :])])
    return b_range


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
    for i in np.arange(1, f + 1):
        plt.figure(i)
        plt.clf()


'''
MAIN
'''

seed = 1
np.random.seed(seed)

'''
Figure 22. figure entropy production rate as a function of irreversibility (theoretical)
'''
d = 2  # dimension
n = 20  # number of epr computations
#x0 = np.array([0, 0])  # initial condition
#epsilon = 0.01  # time-step

B1 = np.eye(d)  # symmetric matrix
B2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
sigma = np.eye(d)  # volatility matrix

epr = np.zeros([n])  # epr
deltas = np.linspace(0, 1 - 0.01, n)
i = 0  # set counter
for delta in deltas:
    process = OU_process(dim=d, friction=B1 + delta * B2, volatility=sigma)
    epr[i] = ent_prod_rate_2D(process)
    i += 1

plt.figure(22)
plt.clf()
plot_cool_colourline2(deltas, epr, deltas, lw=1)

plt.suptitle("Entropy production rate (theoretical)")

plt.xlabel('delta')
plt.ylabel('epr')
# plt.savefig("22.OUprocess.epr(delta).png")
# plt.clf()


'''
Figure 29. figure entropy production rate as a function of irreversibility (Monte Carlo)
'''
print("Starting Figure 29")
N = 10  # number of Monte carlo estimates (ie trajectories
#T = 4  # number of timesteps

# B1 = np.eye(d)  # symmetric matrix
# B2 = np.array([0, 1, -1, 0]).reshape([2, 2])  # antisymmetric matrix
# sigma = np.eye(d)  # volatility matrix
D = 0.5 * sigma * sigma.T  # diffusion tensor

epr_mc = np.empty([n])  # epr monte carlo estimate
deltas = np.linspace(0, 1 - 1e-10, n)
i = 0  # set counter

for delta in deltas:
    B = B1 + delta * B2
    process = OU_process(dim=d, friction=B, volatility=sigma)
    #x = process.simulation(x0, epsilon, T, N)
    S = process.stat_cov2D()
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N,1]).T  # generate stationary samples
    if np.linalg.det(S) == 0:
        print(i)
        print(S)
        raise TypeError('Not hypoelliptic')
    if np.linalg.det(D) == 0:
        print(i)
        print(D)
        raise TypeError('Not elliptic')
    Sinv = np.linalg.inv(S)
    Q = B @ S - D
    Dinv = np.linalg.inv(D)
    for sim in range(0, N):
        y = Q @ Sinv @ x[:, -1, sim] #probability flux
        epr_mc[i] += y.T @ Dinv @ y #integrand of entropy prod rate
    epr_mc[i] = epr_mc[i] / N
    i += 1

plt.figure(29)
plt.clf()
plot_cool_colourline2(deltas, epr_mc, deltas, lw=1)

plt.suptitle("Entropy production rate (Monte-Carlo)")

plt.xlabel('delta')
plt.ylabel('epr')

plt.savefig("29.OUprocess.epr(delta).Monte-Carlo.png")

'''
Figure 30. entropy production rate as a function of irreversibility (via instantaneous epr)
'''

print("Starting Figure 30")

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
    B = B1 + delta * B2
    process = OU_process(dim=d, friction=B, volatility=sigma)
    S = process.stat_cov2D()  # stationary covariance
    x = np.transpose(np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N,
                                                                                    1]))  # generate initial condition at steady-state (since known)
    t = 0
    while t < T:
        steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        # pos[:, t:(t + steps), 0, i] = x[:, :, 0]  # record positions
        epr_inst[t:(t + steps - 1), i] = inst_epr(x, epsilon, nbins=10)  # record instantaneous entropy production
        H[t:(t + steps), i] = entropy(x, nbins=10)  # record entropy
        t += steps
    epr_v[i] = np.mean(epr_inst[epr_inst[:, i] >= 0, i])
    epr_v2[i] = np.median(epr_inst[epr_inst[:, i] >= 0, i])
    i += 1

# Plotting epr
plt.figure(30)
plt.clf()
plot_cool_colourline2(deltas, epr_v, deltas, lw=1)

plt.suptitle("Entropy production rate (via instantaneous)")

plt.xlabel('delta')
plt.ylabel('epr')

plt.savefig("30.OUprocess.epr(delta).via_instantaneous.png")

# Plotting epr median
plt.figure(35)
plt.clf()
plot_cool_colourline2(deltas, epr_v2, deltas, lw=1)

plt.suptitle("Entropy production rate (via instantaneous), median")

plt.xlabel('delta')
plt.ylabel('epr')

plt.savefig("35.OUprocess.epr(delta).via_instantaneous.median.png")

# Plotting entropy
plt.figure(31)
plt.clf()

plt.suptitle("Entropy")

plt.xlabel('time')
plt.ylabel('H')
for i in range(n):
    plt.plot(H[:, i])

# plotting instantanous epr
plt.figure(32)
plt.clf()
plt.suptitle("Instantaneous epr")

plt.xlabel('time')
plt.ylabel('Inst epr')
for i in range(n):
    plt.plot(epr_inst[epr_inst[:, i] >= 0, i])

'''
Figure 40. entropy production rate as a function of irreversibility (via Markov chain epr)
'''

print("Starting Figure 40")

N = 10 ** 1  # number of trajectories (used to estimate the law of the process via histogram method)
T = 5 * 10 ** 1  # number of timesteps to run the process over
nbins = 10  # bins per dimension

epr_mp = np.zeros([n])  # epr
epr_mp2 = np.zeros([n])
epr_mp3 = np.zeros([n])
P = np.empty([nbins ** d, nbins ** d])  # empirical transition probability from bin to bin
mu = np.empty([nbins ** d])  # probability of stationary distribution

i = 0  # set counter
for delta in deltas:
    print(np.round(i / n * 100, 2))
    B = B1 + delta * B2
    process = OU_process(dim=d, friction=B, volatility=sigma)
    S = process.stat_cov2D()  # stationary covariance
    x = np.transpose(np.random.multivariate_normal
                     (mean=np.zeros([d]), cov=S,
                      size=[N, 1]))  # generate initial condition at steady-state (since known)
    x = process.simulation(x[:, -1, :], epsilon, T=T, N=N)  # run simulation
    [P, mu] = stationary_trans_density(x, nbins=nbins)  # compute stationary density and transition probabilities

    MuP = np.repeat(mu, nbins ** d).reshape([nbins ** d, nbins ** d]) * P
    e = (MuP > 0)
    if np.count_nonzero(np.isnan(e)) > 0:
        raise TypeError('error 404')
    e = ~(e * e.T)  # scores where at least one of them is zero
    MuP[e] = 1  # these matrices must equal 1 on e
    epr_mp2[i] = np.sum((MuP - MuP.T) * np.log(MuP / MuP.T)) / 2
    for j in range(nbins):
        for k in range(nbins):
            if e[j, k] == 0:
                epr_mp[i] += (mu[j] * P[j, k] - mu[k] * P[k, j]) * np.log((mu[j] * P[j, k]) / (mu[k] * P[k, j]))
            else:
                epr_mp3[i] += (mu[j] * P[j, k] - mu[k] * P[k, j])/2 * np.log((mu[j] * P[j, k]) / (mu[k] * P[k, j]))
    epr_mp[i] = epr_mp[i] / 2
    i += 1

# Plotting
plt.figure(40)
plt.clf()
plot_cool_colourline2(deltas, epr_mp2, deltas, lw=1)

plt.suptitle("Entropy production rate (via Markov chain)")

plt.xlabel('delta')
plt.ylabel('epr')

plt.savefig("40.OUprocess.epr(delta).markov_chain.png")


'''
Compare EPR given by formula 2D, and integral computed via MC
'''
N = 10 ** 3  # number of MC samples
