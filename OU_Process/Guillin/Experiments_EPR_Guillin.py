'''
Imports
'''
import numpy as np
import sympy as sp
import scipy as sc
import OU_Process.Guillin.gramschmidt as gs
import OU_Process.Guillin.EPR_Guillin as eprg
import OU_Process.Functions.OU_process_functions as OU
import matplotlib.pyplot as plt


'''Analysis '''

def curve_fitting():
    plt.figure(82)
    plt.plot(q, epr_v_canon)
    plt.plot(q, epr_v)

    temp = np.diff(epr_v)
    temp2 = np.diff(temp)
    plt.figure(82)
    plt.plot(q[:-1], temp)
    plt.plot(q[:-2], temp2)

    #from -4.7 to 14.5 it is relatively homogeneous
    #these are the indices 17 to 46 in q

    #curve fitting in canon setting
    epr = epr_v[17:47]
    x = q[17:47]
    opt_greedy = [0,0,0] #optimal configuration of a,b,c
    opt_sum = 10**6
    for a in np.linspace(0.1,0.3,100):
        for b in np.linspace(-2.5,2,100):
            for c in np.linspace(5.5,8,100):
                s = np.sum((a*x**2+b*x+c - epr) **2)
                if s< opt_sum:
                    opt_sum=s
                    opt_greedy = [a,b,c]
    plt.figure(85)
    plt.clf()
    plt.plot(x, epr)
    plt.plot(x, opt_greedy[0]*x**2+opt_greedy[1]*x+opt_greedy[2])

    #curve fitting via gradient descent
    epr_c = epr[17:47] #epr_v[17:47]
    x = q[17:47]
    opt = np.zeros(3) #optimal configuration of a,b,c
    epsilon = 10**(-6) #time-step
    s=10
    t=0
    while t <10**5:
        s = opt[0]*x**2+opt[1]*x+opt[2]-epr_c
        opt = opt - epsilon * np.sum(np.array([s*x**2,s*x,s]),axis=1)
        if not np.isfinite(np.sum(s **2)):
            raise TypeError(t)
        else:
            print(np.sum(s **2))
        t=t+1
    plt.figure(86)
    plt.clf()
    plt.plot(x, epr_c)
    plt.plot(x, opt[0]*x**2+opt[1]*x+opt[2])

    plt.figure(72)
    plt.clf()
    plt.plot(x, epr_c)
    plt.plot(x, opt[0]*x**2+opt[1]*x+opt[2])
    plt.suptitle("EPR best fit canonical potential")
    plt.savefig("72.EPR_Guillin.best_fit.canon_pot.png")

    plt.figure(73)
    plt.clf()
    plt.plot(x, epr_c)
    plt.plot(x, opt[0]*x**2+opt[1]*x+opt[2])
    plt.suptitle("EPR best fit quadratic potential 1,2")
    plt.savefig("73.EPR_Guillin.best_fit.q_pot.1,2.png")

    plt.figure(74)
    plt.clf()
    plt.plot(q, epr_v)
    plt.plot(q, opt[0]*q**2+opt[1]*q+opt[2])
    plt.suptitle("EPR best fit quadratic potential 1,2")
    plt.savefig("74.EPR_Guillin.best_fit.q_pot.1,2.unzoom.png")


'''
Results from experiment
'''
S_canon = np.array([1, 0, 0, 1]).reshape([2, 2])  # stationary covariance (SPD)
epr_v_canon = np.array([35.43589247, 32.27298326, 35.55001201, 53.56884181,41.87931741,46.01311788, 31.22106255, 39.92216546 ,35.95529659, 21.10583109, 34.12079384,29.42048467,27.29450911,26.76341771,23.41006775 , 27.93471733,23.22475099,22.90038727,21.73449994,19.28090566 , 16.61021871,14.28713941,12.01029264, 9.79716691,7.85154919, 6.17371813,4.65047567,3.26676534,2.17658686,1.24803702, 0.60676248,0.21711451,0.14329209,  0.42302153,0.94829888 , 1.7259698,2.71022908,4.050054,  5.35223803,6.88958962 , 9.02164201,10.68752538,13.01181809,  15.52208286,17.78626767 , 20.16669986,22.48851789,21.63479106,  25.47725352,25.08580483])
S2 = np.array([1, 0, 0, 2]).reshape([2, 2])  # stationary covariance (SPD)
epr_v= np.array([47.14805553,42.35157243,43.25644082,40.99021202,39.12824542,38.4369093,38.05495285,34.93048952,35.9315434,33.40784607 ,29.19197521,27.30044456,27.85324982,25.84889691,22.08587963,23.63069274,21.29057658,20.72411869,18.76573566,16.83424269,14.80228025,12.74787536,10.79930816,9.04827631,7.36491332,5.84406519,4.59094458,3.34960887,2.30076624,1.42520801,0.77929366,0.31415145,0.10313326,0.21815519,0.58603237,1.18764711,2.00410762,2.97419776,4.12004817,5.42074017,6.88875785,8.53851963,10.30067523,12.0644665,13.99534121,16.27449734, 18.26397033, 20.24366985, 21.73394041, 21.90118327])

#using parameters
I = 50  # number of EPR computations
q = np.linspace(-16.1, 16.5, I)
N = 10 ** 5  # number of trajectories (used to estimate the law of the process via histogram method)
T = 1 * 10 ** 4  # number of timesteps to run the process over
epsilon = 0.01  # time-step in the simulations

'''
Canonical potential case
'''
N = 10 ** 5  # number of trajectories (used to estimate the law of the process via histogram method)

A, sigma, Abis = eprg.optimal_OU_coeffs(S_canon)  # [0:2]  # compute the coeffs of the optimal OU process
B2 = A - np.diag(np.diagonal(A))  # solenoidal perturbation for the potential
B2 = B2 * 2 / np.sum(np.abs(B2))  # normalise
d = np.shape(S_canon)[0]  # dimension of state-space

e_p = np.empty(q.shape)

seed = 1
np.random.seed(seed)
for i in range(I):
    print(f'{np.round(i / I * 100, 2)} %')  # specify how far we are in the simulation
    #print(-A + q[i] * B2)
    process = OU.OU_process(dim=d, friction=-A + q[i] * B2, volatility=sigma)
    e_p[i] = OU.epr_int_MC(process, N)

plt.figure(101)
plt.clf()
plt.plot(q[17:47]-5, e_p[17:47], lw=0.5)
plt.plot(q[17:47]-5, epr_v_canon[17:47]*e_p[18]/epr_v_canon[18], lw=0.5)
plt.suptitle("EPR pseudo-inv vs via inst OU hypo canon quadratic pot")
plt.savefig("101.EPR_pinv_via_inst.hypo.OUprocess.canonV.png")
print(e_p[18]/epr_v_canon[18])



''' Skewed potential case'''

A, sigma, Abis = eprg.optimal_OU_coeffs(S2)  # [0:2]  # compute the coeffs of the optimal OU process
B2 = A - np.diag(np.diagonal(A))  # solenoidal perturbation for the potential
B2 = B2 * 2 / np.sum(np.abs(B2))  # normalise

e_p2 = np.empty(q.shape)
i=0
for i in range(I):
    print(f'{np.round(i / I * 100, 2)} %')  # specify how far we are in the simulation
    #print(-A + q[i] * B2)
    process = OU.OU_process(dim=d, friction=-A + q[i] * B2, volatility=sigma)
    #S2 = process.stat_cov2D()  # stationary covariance
    e_p2[i] = OU.epr_int_MC(process, N)
    i += 1

plt.figure(102)
plt.clf()
plt.plot(q[17:47]-A[1,0]*3/4, e_p2[17:47], lw=0.5)
plt.plot(q[17:47]-A[1,0]*3/4, epr_v[17:47]*e_p2[18]/epr_v[18], lw=0.5)
plt.suptitle("EPR pseudo-inv vs via inst OU hypo skewed 1,2 quadratic pot")
plt.savefig("102.EPR_pinv_via_inst.hypo.OUprocess.skewedV.12.png")
print(e_p2[18]/epr_v[18])
