'''Here, we compare int and empirical computations of epr'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn
import OU_Process.Functions.OU_process_functions as ou
import OU_Process.Guillin.EPR_Guillin as eprG

'''
MAIN
'''
seed = 1
np.random.seed(seed)

d = 2  # dimension
n = 25  # number of epr computations
N = 10 ** 5
b = 10
# x0 = np.array([0, 0])  # initial condition
T = 10 ** 3
epsilon = 10**(-9)  # time-step


'''hypoelliptic'''
B_prime = np.array([0, 1, -3, 0]).reshape([d, d])
sigma = np.array([1, 2, 2, 3]).reshape([d, d])
#A = np.eye(d)
B = np.array([1, 0, 0, 3]).reshape([d, d])


min = 0
max = 25
deltas = np.linspace(min, max, n)
#omegas = np.linspace(min, max, n)

epr_theo = np.empty(n)  # epr theoretical
epr_mc = np.empty(n)  # epr integral form
epr_v_inst = np.empty(n)  # epr measure theoretic formula via inst epr
epr_hypo = np.empty(n)
epr_hypo3107 = np.empty(n)

for i in range(n):
    print(f'{np.round(i / n * 100, 2)}%')
    process = ou.OU_process(dim=d, friction=B + deltas[i] * B_prime, volatility=sigma)

    epr_theo[i] = ou.ent_prod_rate_2D(process)
    #epr_mc[i] = ou.epr_int_MC(process, N)
    epr_hypo[i] = ou.epr_hypo_1207(process,epsilon)
    epr_hypo3107[i] = ou.epr_hypo_3107(process)
    #epr_v_inst[i] = ou.epr_via_inst(process, N, T, epsilon, bins=b)[0]


#epr_mc= np.append(epr_mc,np.flip(epr_mc[:-1]))
#epr_v_inst= np.append(epr_v_inst,np.flip(epr_v_inst[:-1]))
#deltas = np.append(deltas,np.flip(np.abs(deltas[:-1])))


#plt.figure(10)
#plt.clf()
# ou.plot_cool_colourline2(deltas[ind], epr_theo[ind], deltas[ind], lw=1)
# ou.plot_cool_colourline2(deltas[ind], epr_mc[ind], deltas[ind], lw=1)
# ou.plot_cool_colourline2(deltas[ind], epr_v_inst[ind], deltas[ind], lw=1)
plt.figure(10)
plt.clf()
plt.plot(deltas, epr_theo)
plt.plot(deltas,epr_hypo)
plt.plot(deltas,epr_hypo3107)
plt.suptitle('Comparing EPR estimators')
plt.savefig("120.EPR_1207_EPR_theo_ellip.png")

plt.figure(11)
plt.clf()
plt.plot(deltas, epr_theo)
plt.plot(deltas,epr_hypo)
plt.plot(deltas,epr_hypo3107)
#print(epr_theo[-1]/epr_hypo[-1])
#plt.xscale('log')
plt.yscale('log')
plt.suptitle('Comparing EPR estimators (log scale)')
plt.savefig("120.EPR_1207_EPR_theo_ellip_log.png")
#plt.plot(deltas,epr_v_inst* epr_hypo_test[1] / epr_v_inst[1])

print(np.sum(epr_hypo3107-epr_theo))
#plt.figure(100)
#plt.clf()
# plt.plot(deltas, epr_theo, lw=0.5)
#plt.plot(deltas, epr_mc, lw=0.5)
#plt.plot(deltas, epr_mc[1] / epr_v_inst[1] * epr_v_inst, lw=0.5)
#plt.suptitle("EPR pseudo-inv vs EPR via inst OU process")
#plt.savefig("100.EPR_pinv_via_inst.ellipOUprocess.canonV.png")

#print(epr_mc)
#print(epr_v_inst)
#print(epr_mc[1] / epr_v_inst[1])

'''Result
epr_mc =np.array([199.07304229, 184.40280045, 167.82407458, 153.73064416,
       138.75565113, 125.67613046, 112.16413195, 100.24415347,
        89.16952642,  77.37797298,  67.81525517,  58.59220775,
        49.68518935,  41.98075879,  34.52817363,  28.00610987,
        22.19831355,  16.94787836,  12.48842223,   8.61234916,
         5.55629375,   3.13037134,   1.38335719,   0.34805282,
         0.        ,   0.34805282,   1.38335719,   3.13037134,
         5.55629375,   8.61234916,  12.48842223,  16.94787836,
        22.19831355,  28.00610987,  34.52817363,  41.98075879,
        49.68518935,  58.59220775,  67.81525517,  77.37797298,
        89.16952642, 100.24415347, 112.16413195, 125.67613046,
       138.75565113, 153.73064416, 167.82407458, 184.40280045,
       199.07304229])
epr_v_inst = np.array([26.76744422, 25.23144805, 23.28523002, 20.94850217, 19.17235063,
       17.3826195 , 15.52051513, 13.79802683, 12.25007736, 10.82507766,
        9.4051534 ,  8.21873172,  7.01119944,  5.90994501,  5.00075634,
        4.01061013,  3.21580533,  2.52554126,  1.88989644,  1.43284082,
        0.95963264,  0.63342061,  0.40185282,  0.2571035 ,  0.21615709,
        0.2571035 ,  0.40185282,  0.63342061,  0.95963264,  1.43284082,
        1.88989644,  2.52554126,  3.21580533,  4.01061013,  5.00075634,
        5.90994501,  7.01119944,  8.21873172,  9.4051534 , 10.82507766,
       12.25007736, 13.79802683, 15.52051513, 17.3826195 , 19.17235063,
       20.94850217, 23.28523002, 25.23144805, 26.76744422])
'''