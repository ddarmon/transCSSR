from transCSSR_bc import *

import numpy
import scipy

import matplotlib.pyplot as plt

# machine_fname = 'transCSSR_results/+even-exact.dot'
# machine_fname = 'transCSSR_results/+golden-mean.dot'
# machine_fname = 'transCSSR_results/+barnettX.dot'
# machine_fname = 'transCSSR_results/+RnC.dot'
machine_fname = 'transCSSR_results/+RIP-exact.dot'
# machine_fname = 'transCSSR_results/+RIP.dot'
# machine_fname = 'transCSSR_results/+complex-csm.dot'
# machine_fname = 'transCSSR_results/+renewal-process.dot'

axs = ['0', '1']

inf_alg = 'transCSSR'

HLs, hLs, hmu, ELs, E, Cmu, etas_matrix = compute_ict_measures(machine_fname, axs, inf_alg, L_max = 50, to_plot = True)

print('Cmu = {}\nH[X_{{0}}] = {}\nhmu = {}\nE   = {}'.format(Cmu, HLs[0], hmu, E))

def Hp(p):
	x = numpy.array([p, 1 - p])

	return -numpy.sum(x*numpy.log2(x))

p = 0.5
q = 0.5

Hp(1/(2 - p)) - Hp(p)/(2 - p) # E for Golden Mean process

numpy.log2(p + 2) - p*numpy.log2(p)/(p + 2) - (1 - p*q)/(p+2)*Hp((1 - p)/(1 - p*q)) # E for RIP

plt.show()