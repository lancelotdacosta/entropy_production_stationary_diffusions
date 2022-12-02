'''
Functions related to OU processes
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

'''
Process
'''


class OU_process(object):

    def __init__(self, dim, friction, volatility):
        super(OU_process, self).__init__()  # constructs the object instance
        self.d = dim
        self.B = friction #negative drift matrix (its eigenvalues should have strictly positive real parts for the process to have a steady-state
        self.sigma = volatility


    def simulation(self, x0, epsilon=0.01, T=100, N=1):  # run Euler OU process for multiple trajectories
        w = np.random.normal(0, np.sqrt(epsilon), (T - 1) * self.d * N).reshape(
            [self.d, T - 1, N])  # random fluctuations
        w = np.round(w, decimals=14)
        # sqrt epsilon because standard deviation, but epsilon as covariance
        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        for t in range(1, T):
            '''Euler-Maruyama'''
            x[:, t, :] = x[:, t - 1, :] - epsilon * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                         + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)
            x = np.round(x, decimals=14)
            if np.count_nonzero(np.isnan(x)):
                raise TypeError("nan")
        return x

    def exact_simulation(self, x0, epsilon=0.01, T=100,
                         N=1):  # run OU process for multiple trajectories using an exact scheme
        '''
        Covariance of transition kernels
        '''
        Q_eps = np.empty([self.d, self.d])
        for i in range(self.d):
            for j in range(self.d):
                Q_eps[i, j] = scipy.integrate.quad(func=integrand, a=0, b=epsilon, args=(self.B, self.sigma, i, j))[0]
        #print(Q_eps)
        '''
        Samples from N(0, covariance of transition kernels)
        '''
        w = np.random.multivariate_normal(mean=np.zeros(self.d), cov=Q_eps, size=(T - 1) * N).T  # random fluctuations
        w = np.round(w, decimals=14)
        if np.count_nonzero(np.isnan(w)):
            raise TypeError("nan in fluctuations")
        '''
        Computing exponential of the drift
        '''
        exp_B = scipy.linalg.expm(- epsilon * self.B)
        '''
        Setting up initial condition
        '''
        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        '''
        Perform exact simulation
        '''
        for t in range(1, T):
            x[:, t, :] = np.tensordot(exp_B, x[:, t - 1, :], axes=1) + w[:, (t - 1) * N:t * N]
            x = np.round(x, decimals=14)
            if np.count_nonzero(np.isnan(x)):
                print(f'time-step={t}')
                raise TypeError("nan in process")
        return x


    def BBK_simulation(self, x0, epsilon=0.01, T=100, N=1):  #
        '''
        Run OU process for multiple trajectories using an BBK scheme
        This presupposes that the OU process is underdamped Langevin dynamics in a quadratic potential
        and assumes that the first n variables are positions, while the n latter are momenta
        '''
        '''
        Check dimension is even
        '''
        if self.d % 2 == 1:
            raise TypeError('Odd dimensions input to BBK integrator')
        else:
            n = int(self.d / 2)  # dimension of position/momentum space
        '''
        Check whether volatility has the structure it is expected to
        '''
        s = self.sigma[n, n] #should correspond to sqrt(2 *gamma* beta.inv)

        if np.sum(np.abs(self.sigma[n:, n:] - s * np.eye(n))) > 0:
            raise TypeError('Lower right submatrix of volatility is not a multiple of identity')

        if np.sum(np.abs(self.sigma)) > n * np.abs(s):
            raise TypeError('Volatility is non-zero outside of its lower right submatrix')

        if s<=0:
            raise TypeError('Lower right submatrix of volatility is a NEGATIVE multiple of identity')

        '''
        Check whether drift has the structure it is expected to
        '''
        if np.sum(np.abs(self.B[:n, :n])) > 0:
            raise TypeError('Upper left submatrix of drift is non-zero')

        '''
        Random fluctuations
        '''
        w = np.random.normal(0, np.sqrt(epsilon), self.d *(T - 1) * N).reshape(
            [self.d, T - 1, N])  # random fluctuations
        w = np.round(w, decimals=14)

        '''
        Setting up initial condition
        '''
        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        '''
        Perform BBK simulation
        '''
        for t in range(1, T):
            x[n:, t, :] = x[n:, t - 1, :] - epsilon/2 * np.tensordot(self.B[n:,:], x[:, t - 1, :], axes=1) \
                          + s * w[n:, t - 1, :]  # update momenta (1/2)
            x[:n, t, :] = x[:n, t - 1, :] - epsilon * np.tensordot(self.B[:n,n:], x[n:, t, :], axes=1) # update positions
            x[n:, t, :] = x[n:, t, :] - epsilon/2 * np.tensordot(self.B[n:,:], x[:, t, :], axes=1) \
                          + s*w[:n, t - 1, :]  # update momenta (2/2)
            x = np.round(x, decimals=14)
            if np.count_nonzero(np.isnan(x)):
                raise TypeError("nan")
        return x

    def simulation_float128(self, x0, epsilon=0.01, T=100, N=1):  # run Euler OU process for multiple trajectories float 128
        w = np.random.normal(0, np.sqrt(epsilon), self.d*(T - 1) * N).reshape(
            [self.d, T - 1, N])  # random fluctuations
        w = np.round(w, decimals=14)

        x = np.empty([self.d, T, N])  # store values of the process
        if x0.shape == x[:, 0, 0].shape:
            x[:, 0, :] = np.tile(x0, N).reshape([self.d, N])  # initial condition
        elif x0.shape == x[:, 0, :].shape:
            x[:, 0, :] = x0
        else:
            raise TypeError("Initial condition has wrong dimensions")
        redo = 1
        while redo > 0:
            for t in range(1, T):
                x[:, t, :] = x[:, t - 1, :] - epsilon * np.tensordot(self.B, x[:, t - 1, :], axes=1) \
                             + np.tensordot(self.sigma, w[:, t - 1, :], axes=1)
                x = np.round(x, decimals=14)

                if redo == 2 and not np.all(np.isfinite(x)):  # if there is a nan and we have redone the loop
                    raise TypeError("nan")  # returns an error
                elif not np.all(np.isfinite(x)):  # if there is a nan in x
                    x = np.array(x, dtype=np.float128)  # update to float128
                    print('redo')
                    redo = 2  # scores that we redid the loop
                else:
                    redo = 0  # there is no nan/inf in x so we can leave the loop
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

    def transition_kernel(self,x0,epsilon):
        '''
        :param epsilon:
        :return: exact transition kernel
        '''
        '''
        Covariance of transition kernel
        '''
        Q_eps = np.empty([self.d, self.d])
        for i in range(self.d):
            for j in range(self.d):
                Q_eps[i, j] = scipy.integrate.quad(func=integrand, a=0, b=epsilon, args=(self.B, self.sigma, i, j))[0]
        '''
        Mean of transition kernel
        '''
        exp_B = scipy.linalg.expm(- epsilon * self.B)
        mean =  exp_B@x0

        return multivariate_normal(mean=mean, cov= Q_eps)

    def stationary_covariance(self):
        n = self.d
        S = np.empty(n)  # initialise stationary covariance
        for i in range(n):
            for j in range(n):
                S[i, j] = scipy.integrate.quad(func=integrand, a=0, b=np.inf, args=(self.B, self.sigma, i, j))[0]
        return S

    def stat_cov2D(self):  # compute stationary covariance using 2D analytical formula
        if self.d != 2:
            raise TypeError("This code is exclusively for 2D!")
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
                      c ** 2 * u + (detB + a ** 2) * v - 2 * a * c * w]).reshape(2,
                                                                                 2)  # analytical formula for stationary covariance
        return S / ((a + d) * detB)


'''
Functions
'''


def entropy(x, nbins=10):  # estimate entropy of sample
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)
    entropy = np.zeros(T)
    for t in range(T):
        h = np.histogramdd(x[:, t, :].T, bins=nbins, range=b_range)[0]
        entropy[t] = - np.sum(h / N * np.log(h + (h == 0)))
    return entropy + np.log(N)


def inst_epr(x, epsilon, nbins=10):  # compute instantaneous EPR
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)
    b_range = [val for val in b_range for _ in range(2)]  # duplicate each element the list

    inf_epr = np.zeros([T - 1])

    for t in range(T - 1):
        x_t = x[:, t:t + 2, :]  # trajectories at time t, t+1
        x_mt = np.flip(x_t, axis=1)  # trajectories at time t+1, t (time reversal)
        x_t = x_t.reshape([d * 2, N])  # samples from (x_t, x_t+1)
        x_mt = x_mt.reshape([d * 2, N])  # samples from (x_t+1, x_t) (time reversal)
        e = np.histogramdd(x_t.T, bins=nbins, range=b_range)[0]  # law of (x_t, x_t+1) (unnormalised)
        h = np.histogramdd(x_mt.T, bins=nbins, range=b_range)[0]  # law of (x_t+1, x_t) (unnormalised)
        # nonzero = (e != 0)*(h != 0)
        # inf_epr[t]= np.sum(np.where(nonzero, e/N * np.log(e / h), 0)) / epsilon
        nonzero = (e != 0) * (h != 0)  # shows where e and h are non-zero
        zero = (nonzero == 0)  # shows where e or h are zero
        inf_epr[t] = np.sum(
            e / (N * epsilon) * np.log((e * nonzero + zero) / (h * nonzero + zero)))  # 1/epsilon * KL divergence
    return inf_epr  # 1/epsilon * KL divergence


def epr_samples(x, epsilon, nbins=10):  # approximate EPR with samples from stationary process
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)  # determine grid over which we compute transition probabilities
    b_range = [val for val in b_range for _ in range(2)]  # duplicate each element the list

    p = np.zeros([nbins, nbins, nbins, nbins])  # law of (x_t, x_t+1) (unnormalised)
    p_m = np.zeros([nbins, nbins, nbins, nbins])  # law of (x_t+1, x_t) (unnormalised)

    for t in range(T - 1):
        x_t = x[:, t:t + 2, :]  # transition samples from time t, t+1
        x_mt = np.flip(x_t, axis=1)  # transition samples from time t+1, t (time reversal)
        x_t = x_t.reshape([d * 2, N])  # samples from (x_t, x_t+1)
        x_mt = x_mt.reshape([d * 2, N])  # samples from (x_t+1, x_t) (time reversal)
        p_t = np.histogramdd(x_t.T, bins=nbins, range=b_range)[0]  # law of (x_t, x_t+1) (unnormalised)
        p_mt = np.histogramdd(x_mt.T, bins=nbins, range=b_range)[0]  # law of (x_t+1, x_t) (unnormalised)
        p = p + p_t  # law of (x_t, x_t+1) (unnormalised)
        p_m = p_m + p_mt  # law of (x_t+1, x_t) (unnormalised)

    if np.sum(p) != np.sum(p_m):
        print('error')
        print(np.sum(p - p_m))
    nonzero = (p != 0) * (p_m != 0)  # shows where p and p_m are non-zero
    zero = (nonzero == 0)  # shows where p or p_m are zero
    return np.sum(
        p / (np.sum(p) * epsilon) * np.log((p * nonzero + zero) / (p_m * nonzero + zero)))  # 1/epsilon * KL divergence


def epr_via_inst(process, N, T, epsilon, steps=10, bins=10):
    d = process.attributes()[1].shape[1]
    if d == 2:
        S = process.stat_cov2D()  # stationary covariance
    else:
        S = process.stationary_covariance()
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S,
                                      size=[N, 1]).T  # generate initial condition at steady-state (since known)
    t = 0
    epr_inst = -np.ones([T])  # instantaneous entropy production rate
    while t < T:
        # steps = 10  # number of steps in a simulation
        x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
        # pos[:, t:(t + steps), 0, i] = x[:, :, 0]  # record positions
        epr_inst[t:(t + steps - 1)] = inst_epr(x, epsilon,
                                               nbins=bins)  # record instantaneous entropy production
        t += steps
    epr_v = np.mean(epr_inst[epr_inst >= 0])
    epr_v_median = np.median(epr_inst[epr_inst >= 0])
    return epr_v, epr_v_median


def ent_prod_rate_2D(process):  # analytical formula for the 2D epr
    ''' There seems to be a problem with this formula as I have seen it return negative values'''
    # TODO: update with trace formula which is valid in any dimension (and using pinv)
    [B, sigma] = process.attributes()[0:2]
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


def epr_int_MC(process, N):  # integral formula for EPR elliptic process
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    D = sigma @ sigma.T / 2
    if np.linalg.det(D) == 0:
        D_pinv = np.linalg.pinv(D, rcond=10 ** (-15))
        print('Using pseudo-inverse')
    else:
        D_pinv = np.linalg.inv(D)
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    e_p = 0
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N]).T  # generate stationary samples
    J = B - D @ np.linalg.inv(S)
    for i in range(N):
        Jx = J @ x[:, i]
        e_p += Jx.T @ D_pinv @ Jx / N
    return e_p


def epr_int_MC2(process,
                N):  # integral formula for EPR elliptic process (same as before only uses pinv at a different place)
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    D = sigma @ sigma.T / 2
    D_pinv = np.linalg.pinv(D)
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    e_p = 0
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N]).T  # generate stationary samples
    J = D_pinv @ B - np.linalg.inv(S)
    for i in range(N):
        Jx = J @ x[:, i]
        e_p += Jx.T @ D @ Jx / N
    return e_p


'''These work when B = sigma @ A (e.g. elliptic)'''


def epr_BsigmaA(process, hypo=False, A=0):
    B, sigma = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if hypo:
        if np.sum(np.abs(B - sigma @ A)) > 0:
            print(np.abs(B - sigma @ A))
            raise Warning('A incorrectly specified')
    else:
        if np.linalg.det(sigma) == 0:
            raise Warning('Said not hypoelliptic while it is')
        else:
            A = np.linalg.inv(sigma) @ B
    temp = A - sigma.T @ np.linalg.inv(S) / 2
    return 2 * np.trace(S @ temp.T @ temp)


def epr_BsigmaA_MC(process, N, hypo=False, A=0):  # integral formula for EPR elliptic process
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    e_p = 0
    x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S, size=[N]).T  # generate stationary samples
    if hypo:
        if np.sum(sigma @ B != A) > 0:
            print(np.sum(sigma @ B != A))
            raise Warning('A incorrectly specified')
    else:
        if np.linalg.det(sigma) == 0:
            raise Warning('Said hypoelliptic while it is not')
        else:
            A = np.linalg.inv(sigma) @ B
    J = A - sigma.T @ np.linalg.inv(S) / 2
    for i in range(N):
        Jx = J @ x[:, i]
        e_p += 2 * Jx.T @ Jx / N
    return e_p


'''Epr 11.7.2020'''


def epr_hypo_1107(process):
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    D = sigma @ sigma.T / 2
    A = B + 2 * D * np.linalg.inv(S)
    Q = (B @ S - S @ B.T) / 2
    return np.trace(A.T @ D @ Q)


''' Epr 12.7.2020'''


def epr_hypo_1207(process, epsilon):
    [B, sigma] = process.attributes()
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic')
    A = sigma @ sigma.T
    Q_eps = np.empty([d, d])
    rQ_eps = np.empty([d, d])  # of reverse process
    Sinv = np.linalg.inv(S)
    for i in range(d):
        for j in range(d):
            Q_eps[i, j] = scipy.integrate.quad(func=integrand, a=0, b=epsilon, args=(B, sigma, i, j))[0]
            rQ_eps[i, j] = scipy.integrate.quad(func=integrand, a=0, b=epsilon, args=(-(B - A @ Sinv), sigma, i, j))[0]
    detQ = np.linalg.det(Q_eps)
    detrQ = np.linalg.det(rQ_eps)
    if detQ == 0 or detrQ == 0:
        print(np.linalg.det(Q_eps))
        print(np.linalg.det(rQ_eps))
        raise TypeError('Error: transient covariance is singular')
    else:
        rQ_inv = np.linalg.inv(rQ_eps)
    C = scipy.linalg.expm(epsilon * (B - A @ Sinv)) - scipy.linalg.expm(-epsilon * B)
    c_epr = np.trace(rQ_inv @ Q_eps) - d + np.log(detrQ / detQ) + np.trace(S @ C.T @ rQ_inv @ C)
    return c_epr / (2 * epsilon)


def epr_hypo_3107(process):
    [B, sigma] = process.attributes()
    if not np.all(np.linalg.eigvals(B) > 0):
        print(B)
        print(np.linalg.eigvals(B))
        raise TypeError("B not all positive eigenvalues")
    d = B.shape[1]  # dimension
    if d == 2:
        S = process.stat_cov2D()
    else:
        S = process.stationary_covariance()
    if np.linalg.det(S) == 0:
        print(S)
        raise TypeError('Not hypoelliptic detS=0')
    Sinv = np.linalg.inv(S)
    C = S @ B.T @ Sinv
    Ainv = np.linalg.inv(sigma @ sigma.T)
    return np.trace(S @ (B - C).T @ Ainv @ (B - C)) / 2


'''Epr via markov chain #TODO'''


def stationary_trans_density(x, nbins=10):  # return normalised transition matrix and unnormalised stationary density
    # the input is the discretisation of the process
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


def stationary_density(x, nbins=10):  # return unnormalised stationary density
    [d, T, N] = np.shape(x)
    b_range = bin_range(x)

    x = x.reshape([d, N * T])  # samples from (x_t, x_t+1)

    hist = np.histogramdd(x.T, bins=nbins, range=b_range)[0]  # stationary law of x unnormalised

    # hist= hist/np.sum(hist) # stationary law of x normalised to probabilities

    mu = hist.reshape([nbins ** d])
    return mu


'''Auxiliary functions'''


def integrand(t, B, sigma, i, j):  # to compute stationary covariance
    mx = scipy.linalg.expm(-B * t) @ sigma
    mx = mx @ mx.T
    return mx[i, j]


def bin_range(x):  # create a list of minimum and maximum values of x for each dimension of x
    b_range = []
    for d in range(np.shape(x)[0]):
        b_range.append([np.min(x[d, :, :]) - 10 ** (-12), np.max(x[d, :, :]) + 10 ** (-12)])
    return b_range


'''Plotting functions'''

def plot_cool_colourline3(x, y, lw=0.5):
    d= np.arange(len(x))
    c= cm.cool_r((d - np.min(d)) / (np.max(d) - np.min(d)))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i], linewidth=lw)
    plt.show()
    return

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
