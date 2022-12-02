'''
Estimate EPR for Langevin dynamics on canonical quartic potential
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
import OU_Process.Functions.OU_process_functions as ou
import Langevin.Functions.Langevin_functions as lg

'''Potentials'''


def x4(q):
    return 1 / 4 * np.dot(q, q) ** 2


def x2(q):
    return 1 / 2 * np.dot(q, q)


'''Potential gradients'''


def grad_x4(q):  # V(x)=1/4 x^4
    return q * np.dot(q[-1], q[-1])


def grad_x2(q):  # V(x)=1/2 x^2
    return q

def grad_double_well(q):  # V(x)=1/2 x^2
    return grad_x4(q)-grad_x2(q)


'''Code'''

seed = 1
np.random.seed(seed)

# parameters for process
d = 2
gradV = grad_x2
beta_inv = 1

# parameters for simulation
n = 10 #number of simulations
N = 10 ** 5
T = 10 ** 3
epsilon = 0.001
bins = 2
gamma = np.linspace(205, 10**(-2), n)

#storing epr values
epr_v = np.empty(n)
epr_v_median = np.empty(n)
epr_mc = np.empty(n)

#simulation
for i in range(n):
    # percentage of simulations done
    print(f'{np.round(i / n * 100, 2)}%')

    # create Langevin process
    process = lg.Langevin_process(d, gradV, gamma[i], beta_inv)

    # estimate entropy production rate
    epr_v[i], epr_v_median[i], epr_mc[i] = lg.epr_via_inst(process=process, N=N, T=T, epsilon=epsilon, bins=bins)

#printing
print(epr_mc)
print(epr_v)
print(epr_v_median)

#plotting
plt.figure(110)
plt.clf()
# plt.plot(deltas, epr_theo, lw=0.5)
plt.plot(gamma, epr_mc, lw=0.5)
plt.plot(gamma,  epr_mc[1] / epr_v[1]*epr_v, lw=0.5)
plt.plot(gamma, epr_mc[1] / epr_v_median[1]*epr_v_median, lw=0.5)
plt.yscale('log')
#plt.suptitle("EPR pseudo-inv vs EPR via inst OU process")
#plt.savefig("100.EPR_pinv_via_inst.ellipOUprocess.canonV.png")

#print normalisation constants
print(epr_mc[1] / epr_v_median[1])
print(epr_mc[1] / epr_v[1])

'''Results x2
d = 1
gradV = grad_x4
beta_inv = 1
n = 10 
N = 10 ** 5
T = 10 ** 3
epsilon = 0.001
bins = 10
gamma = np.linspace(205, 10**(-2), n)
epr_mc = np.array([4.21878151e-05, 5.31776344e-05, 6.99053460e-05, 9.39849522e-05,
       1.35712538e-04, 2.11595792e-04, 3.69732344e-04, 8.25655089e-04,
       3.14162778e-03, 6.04612212e+01])
epr_v = np.array([1.52851755, 1.56041794, 1.59161991, 1.56193315, 1.47910799,
       1.37806784, 1.27345051, 1.14619343, 0.85841379, 0.05635818])
epr_v_median = np.array([1.3606884 , 1.35439548, 1.33533465, 1.43651409, 1.33040907,
       1.24536684, 1.04829362, 0.94428155, 0.75942459, 0.03655875])
'''

'''Results x4
d = 1
gradV = grad_x4
beta_inv = 1
n = 10 
N = 10 ** 5
T = 10 ** 3
epsilon = 0.001
bins = 10
gamma = np.linspace(205, 10**(-2), n)
epr_mc = np.array([2.25410834e-09, 2.54959319e-09, 3.04455926e-09, 3.32913596e-09,
       4.24498752e-09, 5.16208845e-09, 7.03017902e-09, 1.04242453e-08,
       2.16821583e-08, 1.39287691e-07])
epr_v = np.array([ 3.04450727,  3.03839773,  3.01364099,  3.04522164,  2.93754608,
        2.78604278,  2.67101335,  2.58092506,  2.79349749, 55.04548093])
epr_v_median = np.array([2.89994211, 3.04765869, 2.98099072, 2.96394603, 2.91346553,
       2.73386525, 2.62849715, 2.52918704, 2.68614062, 1.88764035])
'''

'''Results double well
d = 1
gradV = grad_x4
beta_inv = 1
n = 10 
N = 10 ** 5
T = 10 ** 3
epsilon = 0.001
bins = 10
gamma = np.linspace(205, 10**(-2), n)
epr_mc = np.array([1.50664816e-05 1.68661775e-05 1.95787468e-05 2.26244304e-05
 2.71997644e-05 3.40859612e-05 4.55213424e-05 6.83328353e-05
 1.36588701e-04 4.06651531e-02])
epr_v = np.array([ 3.03968402  3.04061174  3.01491437  3.04160428  2.9383215   2.78147193
  2.66978804  2.59731436  2.78415562 55.07720701])
epr_v_median = np.array([2.91709596 3.04358857 2.97703632 2.97632222 2.90750896 2.72033245
 2.62355581 2.53967892 2.6857601  1.7907298 ])
'''

'''Results x2
d = 2
gradV = grad_x2
beta_inv = 1
n = 10 
N = 10 ** 5
T = 10 ** 3
epsilon = 0.001
bins = 2
gamma = np.linspace(205, 10**(-2), n)
epr_mc = np.array([8.42570427e-05 1.06773845e-04 1.39675293e-04 1.89109799e-04
 2.71780741e-04 4.21848913e-04 7.47431527e-04 1.65537403e-03
 6.31692783e-03 1.21390406e+02])
epr_v = np.array([0.74379867 0.77903996 0.80865057 0.77890015 0.8821744  1.01938298
 1.05576736 1.08775354 1.10591862 0.06303987])
epr_v_median = np.array([0.70075801 0.74617579 0.77625534 0.7233823  0.83563976 0.95085806
 1.01896449 1.07247542 1.08761302 0.03253313])
'''
