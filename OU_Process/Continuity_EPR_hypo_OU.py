'''Here, we assess whether the definition of epr can be extended by continuity'''
'''Results show that this is totally discontinuous'''

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
n = 1000  # number of epr computations
N=10**5
#x0 = np.array([0, 0])  # initial condition
#epsilon = 0.01  # time-step

#B = np.eye(d)  # symmetric matrix
#sigma_d = np.array([1,0,1,0]).reshape([2,2])  # volatility matrix (degenerate)
sigma = np.eye(d)

S = np.array([1, 0, 0, 1]).reshape([2, 2])
A, sigma_d, Abis = eprG.optimal_OU_coeffs(S)


min = 10**(-12)
max = 10**(-4)
epsilon = np.geomspace(min,max,n)-min
epr = np.empty(n)
epr_2 = np.empty(n)
det_non_zero = np.empty(n)

for i in range(n):
    print(f'{np.round(i / n * 100, 2)}%')
    sig = (1-epsilon[i])*sigma_d+epsilon[i]*sigma
    if not np.isfinite(np.linalg.slogdet(sig)[1]):
        print(f'det=0, e={i}')

    else:
        print(f'det={np.linalg.det(sig)}')
    process = ou.OU_process(dim=d, friction=-A, volatility=sig)
    #try:
    epr[i] = ou.epr_int_MC(process, N)
    #epr_2[i] = ou.epr_int_MC2(process, N)
    #except:
    #    epr[i] = -1

plt.figure(90)
plt.clf()
ou.plot_cool_colourline2(epsilon, epr, epsilon, lw=1)
plt.suptitle("EPR continuity as a function of sigma")
plt.savefig("90.EPR_pinv_cont_hypoelliptic.canonV.png")
#ou.plot_cool_colourline2(epsilon, epr_2, epsilon, lw=1)
plt.xscale('log')
#plt.yscale('log')

'''
#Result: not continuous
epr=np.array([2.50331051e+01, 1.44520827e+29, 3.54563154e+28, 1.54730703e+28,
       8.51837549e+27, 5.33190810e+27, 3.63794241e+27, 2.64565163e+27,
       1.96721899e+27, 1.53497238e+27, 1.22431302e+27, 9.85905938e+26,
       8.17588782e+26, 6.84219500e+26, 5.77496632e+26, 4.89500173e+26,
       4.26580123e+26, 3.69525805e+26, 3.20365755e+26, 2.83656329e+26,
       2.50476936e+26, 2.22018081e+26, 2.00097829e+26, 1.78720893e+26,
       1.61841517e+26, 1.45753448e+26, 1.31726094e+26, 1.19912996e+26,
       1.09253282e+26, 9.92834493e+25, 9.22152576e+25, 8.38976594e+25,
       7.71305127e+25, 7.10553417e+25, 6.57815952e+25, 6.11570949e+25,
       5.60554859e+25, 5.23388141e+25, 4.85358557e+25, 4.48697155e+25,
       4.22015549e+25, 3.89894860e+25, 3.67923794e+25, 3.39317078e+25,
       3.19417804e+25, 3.00297574e+25, 2.78744027e+25, 2.60965904e+25,
       2.48101779e+25, 2.32314128e+25, 2.16887591e+25, 2.04972220e+25,
       1.92680963e+25, 1.83392741e+25, 1.70922217e+25, 1.62881107e+25,
       1.52611517e+25, 1.44405529e+25, 1.36619612e+25, 1.29271628e+25,
       1.22879843e+25, 1.15183163e+25, 1.09635497e+25, 1.03875154e+25,
       9.87092195e+24, 9.34067151e+24, 8.81054268e+24, 8.37476847e+24,
       7.98555450e+24, 7.58739579e+24, 7.16245816e+24, 6.88538895e+24,
       6.44845543e+24, 6.18548999e+24, 5.87602841e+24, 5.62323006e+24,
       5.37071343e+24, 5.06725159e+24, 4.82338133e+24, 4.58872349e+24,
       4.41336255e+24, 4.17284712e+24, 3.98126370e+24, 3.82377960e+24,
       3.61630815e+24, 3.47473506e+24, 3.31224142e+24, 3.14327274e+24,
       3.04843767e+24, 2.88883188e+24, 2.76485279e+24, 2.64925744e+24,
       2.49937846e+24, 2.39271373e+24, 2.30862943e+24, 2.21069921e+24,
       2.10090300e+24, 2.00777031e+24, 1.92216557e+24, 1.83299040e+24,
       1.77380963e+24, 1.68820498e+24, 1.61281628e+24, 1.55110249e+24,
       1.48160570e+24, 1.41715579e+24, 1.35827692e+24, 1.30560277e+24,
       1.24562318e+24, 1.20270521e+24, 1.13898678e+24, 1.10039815e+24,
       1.04541668e+24, 1.01023106e+24, 9.68520000e+23, 9.36169859e+23,
       8.91260423e+23, 8.58188968e+23, 8.16466152e+23, 7.78235218e+23,
       7.53087300e+23, 7.27885440e+23, 6.91441908e+23, 6.64989573e+23,
       6.41969378e+23, 6.15549495e+23, 5.90495289e+23, 5.62029161e+23,
       5.43047897e+23, 5.22688969e+23, 4.99723406e+23, 4.81909753e+23,
       4.61615319e+23, 4.44228309e+23, 4.27543959e+23, 4.11136325e+23,
       3.93588994e+23, 3.76879582e+23, 3.61013266e+23, 3.46940565e+23,
       3.34923376e+23, 3.20925925e+23, 3.11517238e+23, 2.95753302e+23,
       2.86090531e+23, 2.76127037e+23, 2.62700433e+23, 2.53696646e+23,
       2.46763493e+23, 2.34436670e+23, 2.25517273e+23, 2.15836120e+23,
       2.07602319e+23, 2.01461976e+23, 1.91618921e+23, 1.83858792e+23,
       1.79148032e+23, 1.71608743e+23, 1.65574567e+23, 1.57401098e+23,
       1.52865805e+23, 1.46446199e+23, 1.40476607e+23, 1.35825905e+23,
       1.30541098e+23, 1.25191889e+23, 1.21568508e+23, 1.16030482e+23,
       1.12655707e+23, 1.07695152e+23, 1.03424626e+23, 9.98333609e+22,
       9.55262233e+22, 9.26617067e+22, 8.84447724e+22, 8.53727216e+22,
       8.20204971e+22, 7.88566903e+22, 7.66640066e+22, 7.33948747e+22,
       7.09105662e+22, 6.72915854e+22, 6.55143778e+22, 6.28072452e+22,
       6.04462057e+22, 5.85210299e+22, 5.60984528e+22, 5.36275842e+22,
       5.22806627e+22, 5.00964607e+22, 4.83384988e+22, 4.63715795e+22,
       4.46193605e+22, 4.27806389e+22, 4.10088610e+22, 3.99594540e+22,
       3.83787731e+22, 3.70672993e+22, 3.57203241e+22, 3.39700044e+22,
       3.30345445e+22, 3.18530486e+22, 3.04783653e+22, 2.95017472e+22,
       2.85686185e+22, 2.74627022e+22, 2.62618346e+22, 2.52104440e+22,
       2.43743673e+22, 2.33571202e+22, 2.25456147e+22, 2.17006332e+22,
       2.10772104e+22, 2.01285052e+22, 1.93128474e+22, 1.86030405e+22,
       1.79591860e+22, 1.73908139e+22, 1.66995154e+22, 1.60475395e+22,
       1.54432606e+22, 1.48339132e+22, 1.43410170e+22, 1.38930143e+22,
       1.33301746e+22, 1.28084869e+22, 1.22529415e+22, 1.19508011e+22,
       1.14469664e+22, 1.10370517e+22, 1.06887016e+22, 1.02979803e+22,
       9.88004329e+21, 9.56679152e+21, 9.20197694e+21, 8.86950097e+21,
       8.52786997e+21, 8.21477991e+21, 7.89706325e+21, 7.64316360e+21,
       7.34579887e+21, 7.05962042e+21, 6.83785000e+21, 6.56940301e+21,
       6.32659976e+21, 6.11648132e+21, 5.88644160e+21, 5.65665712e+21,
       5.52053190e+21, 5.24253490e+21, 4.99735002e+21, 4.87092518e+21,
       4.72631910e+21, 4.53443058e+21, 4.32159528e+21, 4.21079562e+21,
       4.03104563e+21, 3.89493535e+21, 3.75245004e+21, 3.60413143e+21,
       3.49782253e+21, 3.39384631e+21, 3.22497262e+21, 3.10931558e+21,
       3.02498850e+21, 2.90556723e+21, 2.78404686e+21, 2.67873780e+21,
       2.57402386e+21, 2.47861785e+21, 2.38827663e+21, 2.30635347e+21,
       2.22725758e+21, 2.13645895e+21, 2.07974716e+21, 1.98144944e+21,
       1.90881633e+21, 1.85357496e+21, 1.78260892e+21, 1.71916708e+21,
       1.66055157e+21, 1.59367459e+21, 1.54094499e+21, 1.48252378e+21,
       1.41914904e+21, 1.37691640e+21, 1.33116855e+21, 1.28445777e+21,
       1.22350913e+21, 1.18538727e+21, 1.14613663e+21, 1.09832813e+21,
       1.06521626e+21, 1.02556902e+21, 9.87818394e+20, 9.57138182e+20,
       9.15678437e+20, 8.80748866e+20, 8.48483284e+20, 8.16408999e+20,
       7.93517497e+20, 7.65319963e+20, 7.29542222e+20, 7.07100093e+20,
       6.80350333e+20, 6.53174972e+20, 6.34899825e+20, 6.05682037e+20,
       5.86929586e+20, 5.66191030e+20, 5.49584389e+20, 5.23646715e+20,
       5.05200402e+20, 4.87720188e+20, 4.71394766e+20, 4.56516639e+20,
       4.36364794e+20, 4.22901019e+20, 4.07590575e+20, 3.91017766e+20,
       3.76990956e+20, 3.57991517e+20, 3.50837172e+20, 3.35783119e+20,
       3.25539847e+20, 3.13885427e+20, 3.01604683e+20, 2.93003082e+20,
       2.78588654e+20, 2.69203636e+20, 2.60046253e+20, 2.49499200e+20,
       2.43381343e+20, 2.32998031e+20, 2.25173896e+20, 2.16164861e+20,
       2.08209033e+20, 2.02324961e+20, 1.95338177e+20, 1.86763304e+20,
       1.79856360e+20, 1.74407531e+20, 1.65852987e+20, 1.60307023e+20,
       1.55309208e+20, 1.50555950e+20, 1.44082778e+20, 1.38368113e+20,
       1.34236649e+20, 1.29246309e+20, 1.24688206e+20, 1.20225672e+20,
       1.15750034e+20, 1.11402317e+20, 1.07462084e+20, 1.03322672e+20,
       9.82986472e+19, 9.61216221e+19, 9.28971193e+19, 8.87681252e+19,
       8.52396981e+19, 8.32981053e+19, 7.98287656e+19, 7.75290368e+19,
       7.34067848e+19, 7.17423147e+19, 6.96359820e+19, 6.63510394e+19,
       6.38955500e+19, 6.16228307e+19, 5.93801318e+19, 5.73221116e+19,
       5.48309458e+19, 5.32382047e+19, 5.12148230e+19, 4.94956194e+19,
       4.72819953e+19, 4.54753999e+19, 4.41353079e+19, 4.23009325e+19,
       4.10819599e+19, 3.96924259e+19, 3.81539776e+19, 3.68782007e+19,
       3.52995640e+19, 3.42402027e+19, 3.30823318e+19, 3.17152446e+19,
       3.08190182e+19, 2.94643851e+19, 2.82839052e+19, 2.72882381e+19,
       2.63360878e+19, 2.54447971e+19, 2.44819516e+19, 2.32298868e+19,
       2.27016649e+19, 2.19987992e+19, 2.12379608e+19, 2.02741731e+19,
       1.96326326e+19, 1.90460256e+19, 1.83697700e+19, 1.74795478e+19,
       1.70167378e+19, 1.62194647e+19, 1.56187046e+19, 1.51676936e+19,
       1.45038461e+19, 1.40432919e+19, 1.35668753e+19, 1.30477148e+19,
       1.25941454e+19, 1.21354325e+19, 1.16627291e+19, 1.12549151e+19,
       1.08874994e+19, 1.04690250e+19, 1.00585467e+19, 9.77163737e+18,
       9.45766431e+18, 9.05278030e+18, 8.71914133e+18, 8.40936144e+18,
       8.13200851e+18, 7.79685646e+18, 7.49397433e+18, 7.21997232e+18,
       6.96684497e+18, 6.72942445e+18, 6.51737868e+18, 6.26331233e+18,
       6.07868825e+18, 5.79439393e+18, 5.57146205e+18, 5.40244180e+18,
       5.21397536e+18, 4.98514998e+18, 4.84132163e+18, 4.65846248e+18,
       4.49122496e+18, 4.31534364e+18, 4.14577244e+18, 4.03225881e+18,
       3.85623570e+18, 3.72493294e+18, 3.58354337e+18, 3.48656269e+18,
       3.32420661e+18, 3.20180129e+18, 3.11217199e+18, 2.98877631e+18,
       2.87758173e+18, 2.78279097e+18, 2.69058535e+18, 2.58146381e+18,
       2.49142223e+18, 2.39201161e+18, 2.32326889e+18, 2.22521138e+18,
       2.16194324e+18, 2.06124017e+18, 1.99134587e+18, 1.92697136e+18,
       1.87375182e+18, 1.77014619e+18, 1.71128714e+18, 1.65883331e+18,
       1.60169783e+18, 1.54288897e+18, 1.48534425e+18, 1.42579803e+18,
       1.37582021e+18, 1.33728182e+18, 1.27822366e+18, 1.23295873e+18,
       1.18335391e+18, 1.15019230e+18, 1.11089255e+18, 1.06312449e+18,
       1.02407564e+18, 9.89476876e+17, 9.56917340e+17, 9.20015649e+17,
       8.84488677e+17, 8.52320156e+17, 8.19168349e+17, 7.94767575e+17,
       7.63684205e+17, 7.39161239e+17, 7.10916943e+17, 6.90122218e+17,
       6.56275895e+17, 6.39720521e+17, 6.10220468e+17, 5.94919928e+17,
       5.71707658e+17, 5.46702679e+17, 5.26962700e+17, 5.08987465e+17,
       4.92723058e+17, 4.74678343e+17, 4.61923776e+17, 4.39759265e+17,
       4.20611090e+17, 4.09156009e+17, 3.93758437e+17, 3.80513163e+17,
       3.63988330e+17, 3.50492211e+17, 3.40224770e+17, 3.27122908e+17,
       3.16253013e+17, 3.04007207e+17, 2.92754407e+17, 2.82463248e+17,
       2.71225035e+17, 2.62276048e+17, 2.51699606e+17, 2.44153347e+17,
       2.33057385e+17, 2.26284209e+17, 2.18446163e+17, 2.10069548e+17,
       2.03212304e+17, 1.95948075e+17, 1.89013146e+17, 1.81609336e+17,
       1.75113865e+17, 1.69409522e+17, 1.63359550e+17, 1.56343583e+17,
       1.51644034e+17, 1.44079345e+17, 1.41103018e+17, 1.34633211e+17,
       1.30321756e+17, 1.25271093e+17, 1.21017146e+17, 1.16895325e+17,
       1.12128101e+17, 1.08299707e+17, 1.04470653e+17, 1.00893962e+17,
       9.67121755e+16, 9.35654558e+16, 9.03843530e+16, 8.71805035e+16,
       8.27843730e+16, 8.03408000e+16, 7.78658480e+16, 7.44812272e+16,
       7.18326679e+16, 6.91484769e+16, 6.78223494e+16, 6.43533151e+16,
       6.22882664e+16, 5.97520300e+16, 5.73570095e+16, 5.60870989e+16,
       5.42239802e+16, 5.18522377e+16, 4.97956584e+16, 4.84169555e+16,
       4.63066776e+16, 4.49697091e+16, 4.26478814e+16, 4.13645839e+16,
       3.98917538e+16, 3.86344748e+16, 3.70264536e+16, 3.55219867e+16,
       3.46430768e+16, 3.33277330e+16, 3.19828865e+16, 3.10153671e+16,
       2.97238122e+16, 2.87225084e+16, 2.76211417e+16, 2.64175455e+16,
       2.54338910e+16, 2.48846682e+16, 2.38875116e+16, 2.29204577e+16,
       2.21869676e+16, 2.13523114e+16, 2.05389545e+16, 1.98128982e+16,
       1.91014843e+16, 1.84072428e+16, 1.77522292e+16, 1.71505972e+16,
       1.64525802e+16, 1.59203037e+16, 1.54762144e+16, 1.47774104e+16,
       1.42893157e+16, 1.36707359e+16, 1.31486076e+16, 1.27430838e+16,
       1.23446111e+16, 1.18740705e+16, 1.15261279e+16, 1.09094244e+16,
       1.06786475e+16, 1.02542334e+16, 9.79131879e+15, 9.47229699e+15,
       9.09531840e+15, 8.74893992e+15, 8.54629739e+15, 8.18920578e+15,
       7.88177217e+15, 7.54974009e+15, 7.33770677e+15, 7.03437103e+15,
       6.75389009e+15, 6.55530532e+15, 6.28360831e+15, 6.11663351e+15,
       5.88382513e+15, 5.64889225e+15, 5.46485939e+15, 5.26060186e+15,
       5.06393683e+15, 4.90534441e+15, 4.68619144e+15, 4.56584661e+15,
       4.38182452e+15, 4.18874351e+15, 4.08461083e+15, 3.92508903e+15,
       3.75471208e+15, 3.62996922e+15, 3.50996936e+15, 3.39846888e+15,
       3.25851589e+15, 3.14580670e+15, 3.03221853e+15, 2.93803577e+15,
       2.81855559e+15, 2.69423714e+15, 2.59668112e+15, 2.49871775e+15,
       2.43065243e+15, 2.33432015e+15, 2.25891730e+15, 2.18148811e+15,
       2.09672327e+15, 2.01378569e+15, 1.94140536e+15, 1.87618604e+15,
       1.81517945e+15, 1.74520379e+15, 1.66890882e+15, 1.62465964e+15,
       1.55604848e+15, 1.50718273e+15, 1.45297900e+15, 1.39357844e+15,
       1.35422746e+15, 1.29640976e+15, 1.25070640e+15, 1.20574769e+15,
       1.15626836e+15, 1.11820894e+15, 1.08772617e+15, 1.03403987e+15,
       1.00424443e+15, 9.65818197e+14, 9.30569143e+14, 8.88094646e+14,
       8.66411166e+14, 8.33148178e+14, 8.04153761e+14, 7.77455036e+14,
       7.43281994e+14, 7.22441666e+14, 6.93334120e+14, 6.70835917e+14,
       6.42839768e+14, 6.20497412e+14, 5.96014076e+14, 5.75709178e+14,
       5.53388114e+14, 5.31550888e+14, 5.17484919e+14, 4.95459226e+14,
       4.78899719e+14, 4.62083310e+14, 4.45489928e+14, 4.27996705e+14,
       4.15902445e+14, 3.97131640e+14, 3.82103145e+14, 3.69848863e+14,
       3.55272657e+14, 3.41533809e+14, 3.31754665e+14, 3.19261339e+14,
       3.05778870e+14, 2.97192536e+14, 2.87222969e+14, 2.76247581e+14,
       2.63617471e+14, 2.57288024e+14, 2.47196331e+14, 2.38093652e+14,
       2.27983817e+14, 2.19779035e+14, 2.13317451e+14, 2.04141757e+14,
       1.96459146e+14, 1.90100142e+14, 1.82133549e+14, 1.77539503e+14,
       1.70281203e+14, 1.63434054e+14, 1.57900987e+14, 1.52846449e+14,
       1.46279996e+14, 1.42066352e+14, 1.36452460e+14, 1.30750107e+14,
       1.26975371e+14, 1.22590664e+14, 1.17928549e+14, 1.13176840e+14,
       1.09554781e+14, 1.05348495e+14, 1.01476696e+14, 9.81118639e+13,
       9.43384126e+13, 9.17097638e+13, 8.82617500e+13, 8.43923444e+13,
       8.15853524e+13, 7.86262823e+13, 7.58890121e+13, 7.30815270e+13,
       7.04538168e+13, 6.75950432e+13, 6.53513965e+13, 6.24052868e+13,
       6.08006122e+13, 5.87100496e+13, 5.59028596e+13, 5.44503644e+13,
       5.25606545e+13, 5.05789894e+13, 4.85557365e+13, 4.69160585e+13,
       4.50796138e+13, 4.34771888e+13, 4.19300131e+13, 4.06995865e+13,
       3.86732679e+13, 3.75551701e+13, 3.60858083e+13, 3.47531514e+13,
       3.39524651e+13, 3.23056589e+13, 3.13009385e+13, 2.98166009e+13,
       2.91397002e+13, 2.77833566e+13, 2.69227015e+13, 2.62211554e+13,
       2.51771237e+13, 2.42772950e+13, 2.30802854e+13, 2.23348211e+13,
       2.16869225e+13, 2.07878334e+13, 2.00617161e+13, 1.92600136e+13,
       1.86744664e+13, 1.80480130e+13, 1.75178799e+13, 1.66832752e+13,
       1.60987390e+13, 1.55564398e+13, 1.49340152e+13, 1.44594744e+13,
       1.39902401e+13, 1.34629860e+13, 1.28951604e+13, 1.24432624e+13,
       1.19380078e+13, 1.14938972e+13, 1.10327455e+13, 1.07113720e+13,
       1.02588579e+13, 9.96631003e+12, 9.67724579e+12, 9.24892850e+12,
       8.92889500e+12, 8.59709290e+12, 8.25665395e+12, 7.97357452e+12,
       7.62599373e+12, 7.38322692e+12, 7.10514846e+12, 6.87802391e+12,
       6.67473726e+12, 6.41138135e+12, 6.16469046e+12, 5.96487672e+12,
       5.73603227e+12, 5.50837774e+12, 5.33353878e+12, 5.08247923e+12,
       4.93518083e+12, 4.82943926e+12, 4.59845178e+12, 4.45202398e+12,
       4.27611097e+12, 4.10113770e+12, 3.94453848e+12, 3.81297247e+12,
       3.65525360e+12, 3.53547227e+12, 3.41684846e+12, 3.29706039e+12,
       3.15468918e+12, 3.07812894e+12, 2.96041225e+12, 2.84130325e+12,
       2.74491556e+12, 2.63917604e+12, 2.54231842e+12, 2.44912984e+12,
       2.36082990e+12, 2.27840645e+12, 2.19049636e+12, 2.12379294e+12,
       2.04223020e+12, 1.95554539e+12, 1.89842408e+12, 1.81179428e+12,
       1.76031372e+12, 1.70302448e+12, 1.63126747e+12, 1.57137809e+12,
       1.51076085e+12, 1.46495950e+12, 1.41425165e+12, 1.35401504e+12,
       1.30275647e+12, 1.26397325e+12, 1.22166389e+12, 1.16370015e+12,
       1.12611975e+12, 1.08622102e+12, 1.04765520e+12, 1.01114909e+12,
       9.74541026e+11, 9.36430167e+11, 9.07005496e+11, 8.76913015e+11,
       8.39819330e+11, 8.07061598e+11, 7.79478993e+11, 7.54045944e+11,
       7.24070272e+11, 7.06417393e+11, 6.76585732e+11, 6.48968371e+11,
       6.24191131e+11, 6.06254032e+11, 5.84646826e+11, 5.59449561e+11,
       5.38036933e+11, 5.20711736e+11, 5.02565425e+11, 4.84789288e+11,
       4.66419502e+11, 4.49501141e+11, 4.32216587e+11, 4.14792329e+11,
       4.02786555e+11, 3.91748270e+11, 3.72408772e+11, 3.60155924e+11,
       3.46845553e+11, 3.33768599e+11, 3.24463909e+11, 3.13189314e+11,
       2.96172755e+11, 2.88406275e+11, 2.77601904e+11, 2.67620018e+11,
       2.58514007e+11, 2.47705057e+11, 2.40428910e+11, 2.31697558e+11,
       2.23289719e+11, 2.12898488e+11, 2.06512053e+11, 1.98100563e+11,
       1.91618037e+11, 1.84678173e+11, 1.78338325e+11, 1.72994963e+11,
       1.66547791e+11, 1.59774734e+11, 1.55760650e+11, 1.48198156e+11,
       1.43849607e+11, 1.38785631e+11, 1.32571163e+11, 1.29998239e+11,
       1.22340484e+11, 1.19361181e+11, 1.15297945e+11, 1.10171845e+11,
       1.06701785e+11, 1.03034324e+11, 1.00084024e+11, 9.55958583e+10,
       9.18635873e+10, 8.78827054e+10, 8.61358886e+10, 8.28704254e+10,
       7.98445051e+10, 7.62719338e+10, 7.39972601e+10, 7.09950871e+10,
       6.84912792e+10, 6.58336994e+10, 6.36835290e+10, 6.07418826e+10,
       5.91934625e+10, 5.70504282e+10, 5.49321851e+10, 5.28885947e+10,
       5.13107067e+10, 4.91282889e+10, 4.68810213e+10, 4.59286503e+10,
       4.39498849e+10, 4.24502968e+10, 4.08090119e+10, 3.92767088e+10,
       3.79814601e+10, 3.65126404e+10, 3.51308307e+10, 3.41617523e+10,
       3.28943177e+10, 3.14429143e+10, 3.07767806e+10, 2.90831225e+10,
       2.81812647e+10, 2.73197692e+10, 2.61579374e+10, 2.53431219e+10,
       2.43337985e+10, 2.35662169e+10, 2.25456388e+10, 2.19869722e+10,
       2.08590971e+10, 2.02729934e+10, 1.94916438e+10, 1.86696197e+10,
       1.81551370e+10, 1.73723805e+10, 1.68990996e+10, 1.62465656e+10,
       1.56430804e+10, 1.50764630e+10, 1.45125250e+10, 1.40241332e+10,
       1.36526742e+10, 1.29452944e+10, 1.25102794e+10, 1.20866449e+10,
       1.16928240e+10, 1.12802037e+10, 1.08904611e+10, 1.03767466e+10,
       1.00787503e+10, 9.64835626e+09, 9.30407784e+09, 9.00512481e+09,
       8.65606710e+09, 8.38604743e+09, 8.10043291e+09, 7.86459637e+09,
       7.55037171e+09, 7.21392962e+09, 7.00237996e+09, 6.70133069e+09,
       6.47642910e+09, 6.19868136e+09, 5.99336340e+09, 5.76101667e+09,
       5.58930178e+09, 5.35156073e+09, 5.16894197e+09, 5.01497920e+09])
'''

