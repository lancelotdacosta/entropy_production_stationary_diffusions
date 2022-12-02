'''Here, we estimate the trace limit in the hypoelliptic case'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.linalg import expm
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
import OU_Process.Functions.OU_process_functions as ou
import OU_Process.Guillin.EPR_Guillin as eprG


'''Set up parameters of OU process'''
d=2
S = np.array([2, 1, 1, 2]).reshape([d, d])  # stationary covariance (SPD)
#S = np.diag([6, 5, 4, 3, 3, 5, 7])

'''Sigma ,B'''
#[B, sigma, Bbis] = eprG.optimal_OU_coeffs(S)
#B=-B
B = np.array([1, 6, 0, 1]).reshape([d, d])
sigma = np.array([1, 0, 1, 0]).reshape([d, d])

''''''
process = ou.OU_process(dim=d, friction=B, volatility=sigma)
S2 = process.stat_cov2D()  # stationary covariance
# S3 = process.stationary_covariance() #stationary covariance
if np.sum(np.abs(S - S2)) > 10 ** (-5):
    print(np.abs(S - S2))
    print("Bad approximation")
if np.linalg.det(S2)==0:
    raise TypeError("Non-invertible")


'''
process = ou.OU_process(dim=d, friction=-Bbis, volatility=sigma)
S2 = process.stat_cov2D()  # stationary covariance
# S3 = process.stationary_covariance() #stationary covariance
if np.sum(np.abs(S - S2)) > 10 ** (-5):
    print(np.abs(S - S2))
    print("Bad approximation")
'''

def trace_limit(B, sigma, S, epsilons):
    d = np.shape(B)[0]
    N = len(epsilons)
    out = np.empty(N)
    Sinv = np.linalg.inv(S)
    for i in range(N):
        e = epsilons[i]
        Q = np.empty([d, d])
        for j in range(d):
            for k in range(d):
                Q[j, k] = scipy.integrate.quad(func=ou.integrand, a=0, b=e, args=(S@ B.T @Sinv, sigma, j, k))[0]
        #print(np.linalg.det(Q))
        if np.linalg.det(Q) !=0:
            Qinv = np.linalg.inv(Q)
            ExpB = expm(-e * B)
            E = S @ ExpB.T @ Sinv - ExpB
            out[i] = np.trace(E.T @ Qinv @ E)
        else:
            out[i]=-1
    return out

def logdet_limit(B, sigma, S, epsilons):
    d = B.shape[1]  # dimension
    N = len(epsilons)
    out = np.empty(N)
    Sinv = np.linalg.inv(S)
    for i in range(N):
        e = epsilons[i]
        Q = np.empty([d, d])
        rQ = np.empty([d, d])  # of reverse process
        for j in range(d):
            for k in range(d):
                Q[j, k] = scipy.integrate.quad(func=ou.integrand, a=0, b=e, args=(B, sigma, j, k))[0]
                rQ[j, k] = scipy.integrate.quad(func=ou.integrand, a=0, b=e, args=(S@ B.T @Sinv, sigma, j, k))[0]

        if np.linalg.det(rQ) !=0:
            rQinv = np.linalg.inv(rQ)
            out[i] = np.trace(rQinv @ Q)
        else:
            out[i]=-1
    return out


epsilons = np.geomspace(10**(-20), 1,50)

result = trace_limit(B,sigma,S2,epsilons)
#result = logdet_limit(B, sigma, S2,epsilons)
print(result)
res_e = result / epsilons

plt.clf()
plt.plot(epsilons[result >=0], result[result >=0])
plt.xscale('log')
plt.yscale('log')
