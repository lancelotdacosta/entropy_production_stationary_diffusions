'''
IMPORTS
'''

import numpy as np
import matplotlib.pyplot as plt

'''
FUNCTIONS
'''


def V(x):
    return x**4-x**2+1 #double well potential

def gradV(x):
    return 4*x**3-2*x

def OLD_V(x0,epsilon,T,beta =1): #run overdamped Langevin dynamics
    w = np.random.normal(0, np.sqrt(epsilon), T)  # random fluctuations
    x = np.zeros(T)  # store values of the process
    x[0] = x0  # initial condition
    for t in range(1, T):
        x[t] = x[t - 1] - epsilon * gradV(x[t - 1]) + np.sqrt(2 * beta ** (-1)) * w[t - 1]
    return x

def OLD_V_vectorised(x0,epsilon,N,T,beta =1): #run overdamped Langevin dynamics for multiple trajectories simultaneously
    w = np.random.normal(0, np.sqrt(epsilon), T * N).reshape([N, T])  # random fluctuations
    x=np.zeros([N,T])  # store values of the process
    x[:,0] = x0  # initial condition
    for t in range(1, T):
        x[:,t] = x[:,t - 1] - epsilon * gradV(x[:,t - 1]) + np.sqrt(2 * beta ** (-1)) * w[:,t - 1]
    return x


'''
MAIN
'''

'''
Setting parameters
'''

seed = 1
np.random.seed(seed)

T = 10**5  # number of time-steps
epsilon = 10 ** (-3)  # amplitude of time-step
beta = 10  # inverse temperature
x0=1 #initial condition

'''
Overdamped Langevin dynamics in 1D
'''

plt.figure(1)
plt.clf()
plt.suptitle("Overdamped Langevin dynamics trajectory")

x = OLD_V(x0,epsilon, T, beta)

plt.xlabel('time')
plt.ylabel('X_t')
plt.plot(range(T), x)

plt.savefig("MetastabilityDynamics1D.png")

plt.figure(2)
plt.clf()
plt.suptitle("Overdamped Langevin dynamics trajectory (2)")
v=np.linspace(-np.abs(x0+0.1),np.abs(x0+0.1),100)
plt.xlabel('X')
plt.ylabel('V(X)')
plt.plot(v, V(v))
plt.scatter(x,V(x),s=0.1,c = 'red')

plt.savefig("ScatterDynamics1D.png")

'''
Estimation of average potential (int V p dx) at each time-step via Monte-Carlo
'''
plt.figure(3)
plt.clf()
plt.suptitle("Expected Potential")

T= 10**3 #number of time-steps in the Monte-Carlo simulations

N = 10 ** 5  # number of Monte-Carlo trajectories

x= OLD_V_vectorised(x0,epsilon,N,T,beta) #run OLD for many trajectories

EV = np.mean(V(x),0) #Estimates potential for each time-step at each trajectory

plt.xlabel('time')
plt.ylabel('E[V(x)]')
plt.plot(range(T), EV) #plots estimated potential

plt.savefig("Potential1D.png")

'''
Estimation of entropy at each time-step
'''

plt.figure(4)
plt.clf()
plt.suptitle("Estimated entropy")

#at each time-step, must take the histogram of all instances of the process and estimate the entropy from it

B= 10**2 #number of bins in the histogram to estimate the entropy

min_x = np.min(x)
max_x= np.max(x)

bins = np.linspace(min_x, max_x, B)
digitized = np.digitize(x, bins) #assigns each datapoint to its corresponding bin

p = np.zeros(T)
entropy = np.zeros(T)
for b in range(B):
    p = np.sum(digitized == b + 1, 0)/N #probability of being in bin b at each time-step
    entropy -= p * np.log(p + (p == 0))

plt.xlabel('time')
plt.ylabel('H[p(x)]')
plt.plot(range(T), entropy)

plt.savefig("Entropy1D.png")


'''
Estimation of free energy (F(p)) at each time-step
'''

plt.figure(5)
plt.clf()
plt.suptitle("Estimated free energy")

plt.xlabel('time')
plt.ylabel('F[p(x)]')
plt.plot(range(T), EV-beta**(-1)*entropy)

plt.savefig("FreeEnergy1D.png")


'''
Overdamped Langevin dynamics in 2D
'''

