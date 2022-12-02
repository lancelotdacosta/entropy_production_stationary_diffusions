'''
We show exact simulations of 2D OU processes
In the reversible, elliptic non-reversible, hypoelliptic and degenerate case
The goal being to show the difference between the three cases
'''

from OU_Process.Functions import OU_process_functions as OU
import numpy as np
from numpy.linalg import inv, det, pinv, matrix_rank
from numpy.linalg import eigvals as spec
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def KRC(B, D):  # test Kalman rank condition (i.e., Hormander's condition for OU process) from drift and diffusion
    # KRC holds iff forall v eigenvector of B.T we have Dv != 0
    # KRC doesn't hold iff exists v eigenvector of B.T with Dv==0
    B_eig = np.linalg.eig(B.T)[1]
    tol = 10 ** (-6)
    KRC = True
    for i in range(B_eig.shape[1]):
        if np.all(np.abs(D @ B_eig[:, i]) < tol):
            KRC = False
    return KRC

'''
Setting up steady-state
'''
dim = 2 #dimension
Pi = np.eye(dim) #precision matrix

'''
Parameters of simulations
'''
x0 = np.ones([dim])/2 #initial condition of simulations
T = 4*10**4 #2*10**4 #number of timesteps
N=2 #number of trajectories per simulation
epsilon = 10**(-4) #timestep

'''
Parameter of plots
'''
cmap_Gaussian = 'Greys'
#n = 0  # which sample path to show (between 0 and N-1)

lim_x = 3
lim_y = 3

x_tick = np.linspace(-lim_x, lim_x, 105)  # x axis points
y_tick = np.linspace(-lim_y, lim_y, 100)  # y axis points
#x_tick = np.linspace(np.min(x[0, :, n]) - 0.5, np.max(x[0, :, n]) + 0.5, 105)  # x axis points
#y_tick = np.linspace(np.min(x[1, :, n]) - 0.5, np.max(x[1, :, n]) + 0.5, 100)  # y axis points

X,Y = np.meshgrid(x_tick,y_tick)
pos = np.dstack((X, Y))

rv = multivariate_normal(cov= inv(Pi)) #random normal

'''
Figure 1: elliptic reversible case
'''
print('Simulation 1: elliptic, reversible')
#setting up process
Q= np.zeros(dim)
sigma = np.eye(dim)
D = sigma @ sigma.T / 2
B = (D + Q) @ Pi  # drift matrix
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)

#performing simulation
np.random.seed(4)
x = process.exact_simulation(x0, epsilon, T, N)

'''
Plotting elliptic reversible case
'''
#plotting result of simulation
print('Starting Figure 1: elliptic, reversible')


fig = plt.figure(1)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)  # plotting the free energy landscape
cmap = plt.cm.get_cmap('cool')
for t in range(1, T):
    if t %10**3 ==0:
        print(t)
    plt.plot(x[0, t-1:t+1, 0], x[1, t-1:t+1, 0], c=cmap(1-t/T), lw = 0.3)
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
plt.title('Elliptic reversible diffusion')
plt.savefig("OUprocess.2D.reversible.paths.png")


'''
Figure 2: elliptic non-reversible case
'''
print('Simulation 2: elliptic, irreversible')
#setting up process
Q= np.array([0, 7, -7, 0]).reshape([dim, dim])
sigma = np.eye(dim)
D = sigma @ sigma.T / 2
B = (D + Q) @ Pi  # drift matrix
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)

#performing simulation
np.random.seed(4)
x = process.exact_simulation(x0, epsilon, T, N)

#plotting result of simulation
print('Plotting Figure 2: elliptic, irreversible')
fig = plt.figure(2)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)  # plotting the free energy landscape
cmap = plt.cm.get_cmap('cool')
for t in range(1, T):
    if t %10**3 ==0:
        print(t)
    plt.plot(x[0, t-1:t+1, 0], x[1, t-1:t+1, 0], c=cmap(1-t/T), lw = 0.3)
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
plt.title('Elliptic irreversible diffusion')
plt.savefig("OUprocess.2D.elliptic-irrev.paths.png")



'''
Figure 3: hypoelliptic case
'''
print('Simulation 3: hypoelliptic, irreversible')
#setting up process
Q= np.array([0, 7, -7, 0]).reshape([dim, dim])
sigma = np.array([1,0,1,0]).reshape([dim, dim])
D = sigma @ sigma.T / 2
B = (D + Q) @ Pi  # drift matrix
print(f'Verifying hypoelliptic... KRC={KRC(B,D)}')
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)

#performing simulation
np.random.seed(4)
x = process.exact_simulation(x0, epsilon, T, N)

#plotting result of simulation
print('Plotting Figure 3: hypoelliptic, irreversible')
fig = plt.figure(3)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)  # plotting the free energy landscape
cmap = plt.cm.get_cmap('cool')
for t in range(1, T):
    if t %10**3 ==0:
        print(t)
    plt.plot(x[0, t-1:t+1, 0], x[1, t-1:t+1, 0], c=cmap(1-t/T), lw = 0.3)
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
plt.title('Hypoelliptic irreversible diffusion')
plt.savefig("OUprocess.2D.hypoelliptic.paths.png")


'''
Figure 4. Degenerate case
'''
print('Simulation 4: degenerate, reversible')

#setting up process
Q= np.zeros([dim,dim])
sigma = np.array([1,0,1,0]).reshape([dim, dim])
D = sigma @ sigma.T / 2
B = (D + Q) @ Pi  # drift matrix
print(f'Verifying not hypoelliptic... KRC={KRC(B,D)}')
process = OU.OU_process(dim=dim, friction=B, volatility=sigma)

#performing simulation
T= int(T/4)
np.random.seed(4)
x = process.exact_simulation(x0, epsilon, T, N)

#plotting result of simulation
print('Plotting Figure 4: degenerate, reversible')
fig = plt.figure(4)
plt.clf()
plt.title('')
plt.contourf(X, Y, rv.pdf(pos), levels=100, cmap=cmap_Gaussian)  # plotting the free energy landscape
cmap = plt.cm.get_cmap('cool')
for t in range(1, T):
    if t %10**3 ==0:
        print(t)
    plt.plot(x[0, t-1:t+1, 0], x[1, t-1:t+1, 0], c=cmap(1-t/T), lw = 0.3)
plt.scatter(x0[0], x0[1], marker='o', c='red', s =20)
plt.text(x0[0], x0[1],s='$\mathbf{x_0}$')
plt.title('Degenerate reversible diffusion')
plt.savefig("OUprocess.2D.degenerate.paths.png")
