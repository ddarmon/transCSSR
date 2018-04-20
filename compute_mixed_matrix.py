from transCSSR import *

import ipdb

import matplotlib.pyplot as plt

# machine_fname = 'transCSSR_results/+even-exact.dot'
# machine_fname = 'transCSSR_results/+golden-mean.dot'
machine_fname = 'transCSSR_results/+RnC.dot'

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

			# print etas_new

	etas_cur = etas_new[1:, :].copy()
	etas_new = numpy.matrix([numpy.nan]*len(M_states_to_index))

	# ipdb.set_trace()

print etas_matrix

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

# plt.figure()
# plt.imshow(W_x['0'])

# plt.figure()
# plt.imshow(W_x['1'])

# plt.figure()
# plt.imshow(W)

D, P = numpy.linalg.eig(W)

HWA = -numpy.nansum(numpy.multiply(numpy.log2(W),W), 1).T

# Is = numpy.matrix(numpy.ones(etas_matrix.shape[0])).T
# HWA = numpy.matrix(numpy.zeros(etas_matrix.shape[0]))

# for mixed_ind in range(etas_matrix.shape[0]):
# 	delta_eta = numpy.matrix(numpy.zeros(etas_matrix.shape[0]))
# 	delta_eta[0, mixed_ind] = 1.

# 	cur_sum = 0.

# 	for x in axs:
# 		p = delta_eta*W_x[x]*Is
# 		if p == 0.:
# 			pass
# 		else:
# 			cur_sum += p*numpy.log2(p)

# 	HWA += delta_eta*float(cur_sum)

# HWA = -HWA

arg_eig_1 = (numpy.isclose(D, 1.)).nonzero()[0][0]
v_eig_1   = P[:, arg_eig_1].T
v_eig_1   = v_eig_1/numpy.sum(v_eig_1)

hmu = v_eig_1*HWA.T

delta_eta = numpy.matrix(numpy.zeros(etas_matrix.shape[0]))
delta_eta[0, 0] = 1.

Wprod = numpy.matrix(numpy.eye(etas_matrix.shape[0]))

hLs = []

for L in range(100):
	hLs.append(float(delta_eta*Wprod*HWA.T))
	print(hLs[-1])

	Wprod = Wprod*W

print('\n')

print(hmu)

plt.plot(hLs)

plt.show()