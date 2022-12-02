'''Here, we compare int and empirical computations of epr'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
import OU_Process.Functions.OU_process_functions as ou
import OU_Process.Guillin.EPR_Guillin as eprg

'''
MAIN
'''
seed = 1
np.random.seed(seed)

d = 2  # dimension
N = 10 ** 5
#b = 10
# x0 = np.array([0, 0])  # initial condition
T = 10 ** 2
epsilon = 10**(-2)  # time-step


'''hypoelliptic'''
B_prime = np.array([0, 1, -1, 0]).reshape([d, d])
sigma = np.array([1, 0, 0, 1]).reshape([d, d])
#A = np.eye(d)
B = np.array([1, 0, 0, 0]).reshape([d, d])

'''Guillin'''
S2 = np.array([1, 0, 0, 2]).reshape([2, 2])  # stationary covariance (SPD)
A, sigma, Abis = eprg.optimal_OU_coeffs(S2)  # [0:2]  # compute the coeffs of the optimal OU process
B2 = A - np.diag(np.diagonal(A))  # solenoidal perturbation for the potential
B2 = B2 * 2 / np.sum(np.abs(B2))  # normalise
d = np.shape(S2)[0]  # dimension of state-space

#I = 50  # number of EPR computations
#q = np.linspace(-16.1, 16.5, I)
#number of processes
I = 5
q = np.linspace(0.1, 10, I)
b=10

#min = 0.1
#max = 25
#deltas = np.linspace(min, max, n)
#omegas = np.linspace(min, max, n)

#epr_v_canon = np.array([35.43589247, 32.27298326, 35.55001201, 53.56884181,41.87931741,46.01311788, 31.22106255, 39.92216546 ,35.95529659, 21.10583109, 34.12079384,29.42048467,27.29450911,26.76341771,23.41006775 , 27.93471733,23.22475099,22.90038727,21.73449994,19.28090566 , 16.61021871,14.28713941,12.01029264, 9.79716691,7.85154919, 6.17371813,4.65047567,3.26676534,2.17658686,1.24803702, 0.60676248,0.21711451,0.14329209,  0.42302153,0.94829888 , 1.7259698,2.71022908,4.050054,  5.35223803,6.88958962 , 9.02164201,10.68752538,13.01181809,  15.52208286,17.78626767 , 20.16669986,22.48851789,21.63479106,  25.47725352,25.08580483])
epr_v= np.array([47.14805553,42.35157243,43.25644082,40.99021202,39.12824542,38.4369093,38.05495285,34.93048952,35.9315434,33.40784607 ,29.19197521,27.30044456,27.85324982,25.84889691,22.08587963,23.63069274,21.29057658,20.72411869,18.76573566,16.83424269,14.80228025,12.74787536,10.79930816,9.04827631,7.36491332,5.84406519,4.59094458,3.34960887,2.30076624,1.42520801,0.77929366,0.31415145,0.10313326,0.21815519,0.58603237,1.18764711,2.00410762,2.97419776,4.12004817,5.42074017,6.88875785,8.53851963,10.30067523,12.0644665,13.99534121,16.27449734, 18.26397033, 20.24366985, 21.73394041, 21.90118327])

#number of epsilons
J=10
epr_hypo = np.empty([I,J])
epr_v_inst = np.empty([I,J])
epsilons = np.geomspace(10**(-14),10**(-2),J)

for i in range(I):
    for j in range(J):
        print(f'{np.round(i / I * 100, 2)}%')
        process = ou.OU_process(dim=d, friction=- q[i] * A, volatility=sigma)

        #epr_theo[i] = ou.ent_prod_rate_2D(process)
        #epr_mc[i] = ou.epr_int_MC(process, N)
        epr_hypo[i,j] = ou.epr_hypo_1207(process, epsilons[j])
        epr_v_inst[i,j] = ou.epr_via_inst(process, N, T, epsilons[j], bins=b)[0]


#epr_mc= np.append(epr_mc,np.flip(epr_mc[:-1]))
#epr_v_inst= np.append(epr_v_inst,np.flip(epr_v_inst[:-1]))
#deltas = np.append(deltas,np.flip(np.abs(deltas[:-1])))


#plt.figure(10)
#plt.clf()
# ou.plot_cool_colourline2(deltas[ind], epr_theo[ind], deltas[ind], lw=1)
# ou.plot_cool_colourline2(deltas[ind], epr_mc[ind], deltas[ind], lw=1)
# ou.plot_cool_colourline2(deltas[ind], epr_v_inst[ind], deltas[ind], lw=1)
plt.figure(1)
plt.clf()
for i in range(I):
    plt.plot(epsilons, epr_v_inst[i,:],lw=0.2)

plt.figure(2)
plt.clf()
for i in range(I):
    plt.plot(epsilons, epr_v_inst[i,:],lw=0.2)
plt.xlabel('epsilon')
plt.ylabel('epr')
plt.xscale('log')
plt.yscale('log')
plt.savefig("130.EPR_v_inst_epsilon_to_0_hypo.png")


plt.figure(3)
plt.clf()
for i in range(I):
    plt.plot(epsilons, epr_hypo[i,:],lw=0.2)

plt.figure(4)
plt.clf()
for i in range(I):
    plt.plot(epsilons, epr_hypo[i,:],lw=0.2)
plt.xlabel('epsilon')
plt.ylabel('epr')
plt.xscale('log')
plt.yscale('log')
plt.savefig("130.EPR_1207_epsilon_to_0_hypo.png")





#plt.figure(10)
#plt.clf()
#plt.plot(q, epr_v)
#plt.plot(q,epr_hypo)

#plt.figure(11)
#plt.clf()
#plt.plot(q, epr_v)
#plt.plot(q,epr_hypo)
#print(epr_theo[-1]/epr_hypo[-1])
#plt.xscale('log')
#plt.yscale('log')
#plt.plot(deltas,epr_v_inst* epr_hypo_test[1] / epr_v_inst[1])


