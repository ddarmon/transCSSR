from transCSSR import *

import numpy
import scipy

import ipdb

import matplotlib.pyplot as plt

# machine_fname = 'transCSSR_results/+even-exact.dot'
# machine_fname = 'transCSSR_results/+golden-mean.dot'
# machine_fname = 'transCSSR_results/+barnettX.dot'
# machine_fname = 'transCSSR_results/+RnC.dot'
# machine_fname = 'transCSSR_results/+RIP.dot'
# machine_fname = 'transCSSR_results/+complex-csm.dot'
machine_fname = 'transCSSR_results/+renewal-process.dot'

axs = ['0', '1']

inf_alg = 'transCSSR'

stationary_dist_eM = None

if stationary_dist_eM == None:
	P, M_states_to_index, M_trans = compute_eM_transition_matrix(machine_fname, axs, inf_alg = inf_alg)

	stationary_dist_mixed, stationary_dist_eM = compute_channel_states_distribution(P, {'A' : 0}, M_states_to_index)

M_index_to_states = {}

for state in M_states_to_index:
	M_index_to_states[M_states_to_index[state]] = state

eta = numpy.matrix(stationary_dist_eM)

Is = numpy.matrix(numpy.ones(len(M_states_to_index))).T

T_x = {}

for x_ind, x in enumerate(axs):
	T_x[x] = numpy.matrix(numpy.zeros((len(M_states_to_index), len(M_states_to_index))))

for S0 in M_states_to_index:
	for x in axs:
		S1, p = M_trans.get((S0, x), (None, 0.0))

		if S1 is not None:
			T_x[x][M_states_to_index[S0], M_states_to_index[S1]] = p

new_states = True

etas_matrix = eta.copy()
etas_cur    = eta.copy()
etas_new = numpy.matrix([numpy.nan]*len(M_states_to_index))

diff_tol = 1e-10

while new_states:
	new_states = False
	for row_ind in range(etas_cur.shape[0]):
		for x in axs:
			new_states = True
			eta = etas_cur[row_ind,:]

			numer = eta*T_x[x]

			eta = numer/(numer*Is)

			if numpy.sum(numpy.isnan(eta)) != len(M_states_to_index):
				print(x, eta)

				diff_dists = numpy.mean(numpy.abs(etas_matrix - eta), 1)

				match_ind = (diff_dists < diff_tol).nonzero()

				# ipdb.set_trace()

				if len(match_ind[0]) == 0: # A new mixed state was generated
					new_states = True

					etas_new = numpy.vstack((etas_new, eta))
					etas_matrix = numpy.vstack((etas_matrix, eta))
				else: # No new mixed state was generated.
					pass

	etas_cur = etas_new[1:, :].copy()
	etas_new = numpy.matrix([numpy.nan]*len(M_states_to_index))

print(etas_matrix)

W_x = {}

for x_ind, x in enumerate(axs):
	W_x[x] = numpy.matrix(numpy.zeros((etas_matrix.shape[0], etas_matrix.shape[0])))

for row_ind in range(etas_matrix.shape[0]):
	eta0 = etas_matrix[row_ind, :]

	for x in axs:
		numer = eta0*T_x[x]

		eta1 = numer/(numer*Is)

		if numpy.sum(numpy.isnan(eta1)) != len(M_states_to_index):
			diff_dists = numpy.mean(numpy.abs(etas_matrix - eta1), 1)

			col_ind = (diff_dists < diff_tol).nonzero()[0]

			W_x[x][row_ind, col_ind] = numer*Is

W = numpy.matrix(numpy.zeros((etas_matrix.shape[0], etas_matrix.shape[0])))

for x in axs:
	W += W_x[x]

row_sums = numpy.array(W.sum(1)).flatten()

for row_ind in range(W.shape[0]):
	W[row_ind, :] = W[row_ind, :]/row_sums[row_ind]


# plt.figure()
# plt.imshow(W_x['0'])

# plt.figure()
# plt.imshow(W_x['1'])

# plt.figure()
# plt.imshow(W)

D, Pl, Pr = scipy.linalg.eig(W, left = True, right = True)

# This only works for eigenvalues that have algebraic multiplicity
# of 1:

# W_lam = {}

# for eigval_ind, eigval in enumerate(D):
# 	eig_right = numpy.matrix(Pr[:, eigval_ind])
# 	eig_left  = numpy.matrix(Pl[:, eigval_ind])

# 	print(eigval, eig_left.T*eig_right)

# 	W_lam[eigval] = (eig_right.T*eig_left)/(eig_left*eig_right.T)

# This only works when W is diagonalizable. (???)

W_lam = {}

Id = numpy.eye(D.shape[0])

# Rounding here causes loss of precision in hmu and E
# D_unique = numpy.unique(D.round(decimals=10))
D_unique = D.copy()

ind_eigval1 = numpy.isclose(D_unique, 1.0).nonzero()[0][0]

for eigval_ind, eigval in enumerate(D_unique):
	W_lam[eigval] = Id.copy()

	for eigval2 in D_unique:
		if eigval == eigval2:
			pass
		else:
			# W_lam[eigval] = ((W - eigval2*Id)/(eigval - eigval2))*W_lam[eigval]
			W_lam[eigval] = W_lam[eigval]*((W - eigval2*Id)/(eigval - eigval2))

HWA = -numpy.nansum(numpy.multiply(numpy.log2(W),W), 1).T

arg_eig_1 = (numpy.isclose(D, 1.)).nonzero()[0][0]
v_eig_1   = Pl[:, arg_eig_1].T
mixed_state_stationary_dist   = numpy.real(v_eig_1/numpy.sum(v_eig_1))

plt.figure()
plt.plot(mixed_state_stationary_dist.T, '.')

hmu = float(mixed_state_stationary_dist*HWA.T)

delta_eta = numpy.matrix(numpy.zeros(etas_matrix.shape[0]))
delta_eta[0, 0] = 1.

hmu2 = float(numpy.real(delta_eta*W_lam[D_unique[ind_eigval1]]*HWA.T))

print('The entropy rates using the stationary distribution over the mixed states and W_{{1}} are:\n{}\n{}'.format(hmu, hmu2))

E = 0.

for eigval in D_unique:
	if numpy.abs(eigval) < 1:
		E += (delta_eta*W_lam[eigval]*HWA.T)/(1 - eigval)

E = float(numpy.real(E))

Wprod = numpy.matrix(numpy.eye(etas_matrix.shape[0]))

hLs = []

L_max = 50

for L in range(L_max):
	hLs.append(float(delta_eta*Wprod*HWA.T))

	Wprod = Wprod*W

hLs = numpy.array(hLs)

ELs = numpy.cumsum(hLs - hmu)

fig, ax = plt.subplots(2)
ax[0].plot(hLs)
ax[0].axhline(hmu, linestyle = '--')
ax[1].plot(ELs)
# ax[1].axhline(E, linestyle = '--')

print('The Excess Entropy E is: {} ({})'.format(E, ELs[-1]))


def Hp(p):
	x = numpy.array([p, 1 - p])

	return -numpy.sum(x*numpy.log2(x))

p = 0.5

Hp(1/(2 - p)) - Hp(p)/(2 - p) # E for Golden Mean process

numpy.log2(p + 2) - p*numpy.log2(p)/(p + 2) - (1 - p*(1-p))/(p+2)*Hp((1 - p)/(1 - p*(1 - p))) # E for RIP

plt.show()