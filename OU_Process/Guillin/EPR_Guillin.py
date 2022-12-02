'''
Compute EPR of optimal OU process from Guillin paper*, and EPR of the same process by adding extra divergence free flow,
to see if the maximum of EPR is also the fastest converging process.
*Guillin, Monmarche - Optimal linear drift for the speed of convergence of an hypoelliptic diffusion
'''

'''
Imports
'''
import numpy as np
import sympy as sp
import scipy as sc
import OU_Process.Guillin.gramschmidt as gs
import OU_Process.Functions.OU_process_functions as OU
import matplotlib.pyplot as plt

''' This function inputs stationary covariance of OU process and outputs drift and volatility for fastest convergence 
The algorithm we use is from
Guillin, Monmarche - Optimal linear drift for the speed of convergence of an hypoelliptic diffusion
'''


def optimal_OU_coeffs(S):
    Sinv = np.linalg.inv(S)  # inverse stationary covariance (used as S in Guillin paper)

    ''' 1. Find eigenvector'''
    eig = np.linalg.eig(Sinv)  # eigenvalues and eigenvectors
    i = np.argmax(eig[0])  # index of maximal eigenvalue
    v = eig[1][i]  # eigenvector of maximal eigenvalue
    ''' 2. Diffusion matrix'''
    D = np.outer(v, v)  # diffusion matrix
    '''Optimal volatility'''
    sigma = np.zeros(np.shape(S))  # initialise volatility
    sigma[:, 0] = np.sqrt(2) * v  # optimal volatility
    '''Sqrt of S^-1'''
    [O, Sinvdiag] = sp.Matrix(Sinv).diagonalize()  # Diagonalisation Sinv = O Sinvdiag O^-1
    O = np.array(O).astype(np.float64)  # convert to numpy array
    Sinvdiag = np.array(Sinvdiag).astype(np.float64)  # convert to numpy array
    sqrtSinvdiag = np.sqrt(Sinvdiag)  # square root of diagonalised Sinv
    sqrtSinv = O @ sqrtSinvdiag @ np.linalg.inv(O)  # obtain sqrt(S^-1)
    ''' 3. Tilde D'''
    tildeD = sqrtSinv @ D @ sqrtSinv
    ''' 4. Orthonormal basis (algo by Lelievre, Nier, Pavliotis, replacing S by tildeD)'''
    N = np.shape(S)[0]
    psi = np.eye(N)  # start arbitrary orthonormal basis
    for n in range(0, N - 1):
        print(n)
        '''4.1 do the permutation'''
        # compute the inner products
        T = psi.T @ tildeD @ psi  # the sought inner products are the diagonal elements of T (temporary matrix)
        diagT = np.diagonal(T)[n:]  # retrieve inner products
        i = np.argmax(diagT) + n  # retrieve index of maximal inner product
        j = np.argmin(diagT) + n  # retrieve index of minimal inner product
        if i == j:  # means all inner products are the same
            j = i + 1
        psi_copy = np.array(psi)  # deep copy of psi
        psi[:, n] = psi_copy[:, i]  # permute columns
        psi[:, n + 1] = psi_copy[:, j]  # permute columns
        psi_copy = np.delete(psi_copy, [i, j], axis=1)  # delete vectors i,j
        psi[:, n + 2:] = psi_copy[:, n:]  # insert the rest to complete the permutation
        '''4.2 compute the optimal combination of psi[:,n], psi[:,n+1] by GD'''
        # greedy search
        t = np.linspace(0, 2 * np.pi, 10 ** 6)  # create a list of values to test for the greedy search
        F_t = T[n, n] * np.cos(t) ** 2 + np.sin(2 * t) * T[n, n + 1] + T[n + 1, n + 1] * np.sin(
            t) ** 2  # compute weighted inner product
        k = np.argmin(np.abs(F_t - np.trace(tildeD) / N))  # index of optimal t
        t_opt = t[k]  # optimal t
        del t, F_t  # free memory
        psi[:, n] = np.cos(t_opt) * psi[:, n] + np.sin(t_opt) * psi[:, n + 1]  # update psi_n by weighting
        '''4.3 Gram-Schmidt to get orthonormal basis of psi_n:'''
        # orth_psi = sc.linalg.orth(psi[:, n:])  # orthogonalised psi
        # the problem is that sc.linalg.orth shuffles vectors. So need to make sure that psi_n remains in the same place
        # instead use the gram-schmidt method
        psi[:, n:] = gs.gramschmidt(psi[:, n:])[0]  # perform orthogonal procedure using gram-schmidt
    '''5. Define P'''
    P = psi / np.sqrt(2)
    '''6. Define hat J'''
    hat_J = np.zeros([N, N])
    for k in range(N):
        for l in range(k + 1, N):
            hat_J[k, l] = -(2 * N + k + l) / (k - l)  # define the upper triangular elements
    hat_J = hat_J - hat_J.T  # define the rest using anti-symmetricity
    '''7. Define tilde J'''
    tilde_J = P @ hat_J @ np.linalg.inv(P)
    '''8. Define the drift A'''
    A = -D @ Sinv - np.linalg.inv(sqrtSinv) @ hat_J @ sqrtSinv  # define drift as is written in the paper
    '''8.bis Define alternative drift'''
    # the problem with the above drift is that it doesn't use tilde_J, while we define it just before
    # hence it is possible that there is a mistake. To avoid this we define an alternative drift that uses the same
    # formula but with tilde J
    Abis = -D @ Sinv - np.linalg.inv(sqrtSinv) @ tilde_J @ sqrtSinv
    return A, sigma, Abis


def EPR_Guillin(S):
    '''Parameters'''
    N = 10 ** 5  # number of trajectories (used to estimate the law of the process via histogram method)
    T = 1 * 10 ** 4  # number of timesteps to run the process over
    epsilon = 0.01  # time-step in the simulations
    I = 50  # number of EPR computations

    A, sigma, Abis = optimal_OU_coeffs(S)  # [0:2]  # compute the coeffs of the optimal OU process
    d = np.shape(S)[0]  # dimension of state-space
    B2 = A - np.diag(np.diagonal(A))  # solenoidal perturbation for the potential
    B2 = B2 * 2 / np.sum(np.abs(B2))  # normalise
    epr_v = np.empty([I])  # epr via instantaneous
    epr_v2 = np.empty([I])  # epr via instantaneous median
    H = -np.ones([T, I])  # score entropy
    epr_inst = -np.ones([T, I])  # instantaneous entropy production rate
    q = np.linspace(-16.1, 16.5, I)  # strength of added solenoidal perturbation
    for i in range(I):
        print(f'{np.round(i / I * 100, 2)} %')  # specify how far we are in the simulation
        print(-A + q[i] * B2)
        process = OU.OU_process(dim=d, friction=-A + q[i] * B2, volatility=sigma)
        S2 = process.stat_cov2D()  # stationary covariance
        # S3 = process.stationary_covariance() #stationary covariance
        if np.sum(np.abs(S - S2)) > 10 ** (-5):
            raise TypeError("Bad approximation")  # If desired and actual covariance are too different
        x = np.random.multivariate_normal(mean=np.zeros([d]), cov=S,
                                          size=[N, 1]).T  # generate initial condition at steady-state (since known)
        t = 0
        while t < T:
            steps = 10  # number of steps in a simulation
            x = process.simulation(x[:, -1, :], epsilon, T=steps, N=N)
            # pos[:, t:(t + steps), 0, i] = x[:, :, 0]  # record positions
            epr_inst[t:(t + steps - 1), i] = OU.inst_epr(x, epsilon,
                                                         nbins=10)  # record instantaneous entropy production
            H[t:(t + steps), i] = OU.entropy(x, nbins=10)  # record entropy
            t += steps
        epr_v[i] = np.mean(epr_inst[epr_inst[:, i] >= 0, i])
        epr_v2[i] = np.median(epr_inst[epr_inst[:, i] >= 0, i])  # estimator via median

    # Plotting epr
    plt.figure(71)
    plt.clf()
    OU.plot_cool_colourline2(q, epr_v, q, lw=1)

    plt.suptitle("Entropy production rate (via instantaneous)")

    plt.xlabel('delta')
    plt.ylabel('epr')

    plt.savefig("71.OUprocess.epr.Guillin.via_instantaneous.canonV.png")

    # Plotting epr median
    plt.figure(76)
    plt.clf()
    OU.plot_cool_colourline2(q, epr_v2, q, lw=1)

    plt.suptitle("Entropy production rate (via instantaneous), median")

    plt.xlabel('delta')
    plt.ylabel('epr')

    plt.savefig("76.OUprocess.epr.Guillin.via_instantaneous.median.canonV.png")

    return epr_v, epr_v2, S


S = np.array([2, 1, 0, 1]).reshape([2, 2])  # stationary covariance (SPD)
S = np.diag([6, 5, 4, 3, 3, 5, 7])
[A, sigma, Abis] = optimal_OU_coeffs(S)
print(A)
print(np.linalg.eig(A)[0])
print(sigma)
[O, Adiag] = sp.Matrix(A).diagonalize()
