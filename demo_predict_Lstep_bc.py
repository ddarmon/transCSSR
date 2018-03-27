import numpy
import scipy.stats
import itertools
import copy
import string
import os

from collections import Counter, defaultdict
from filter_data_methods import *
from igraph import *

from transCSSR_bc import *

import ipdb

data_prefix = ''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# The various test transducers. Xt is the input
# and Yt is the output.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Examples from:
# 		Computational Mechanics of Input-Output Processes- Structured transformations and the epsilon-transducer - Barnett, Crutchfield
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# The z-channel.

# Xt_name = 'Xt_z-channel'
# Yt_name = 'Yt_z-channel'

# The delay-channel.

# Xt_name = 'Xt_delay-channel'
# Yt_name = 'Yt_delay-channel'

# The odd random channel.

# Xt_name = 'Xt_odd-random-channel'
# Yt_name = 'Yt_odd-random-channel'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Additional examples:
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xt_name = 'barnettX'
Yt_name = 'barnettY'

# Xt_name = ''
# Yt_name = 'even'

# Xt_name = 'coinflip'
# Yt_name = 'coinflip-excite_w_refrac'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Set the parameters and associated quantities:
# 	axs, ays -- the input / output alphabets
# 	alpha    -- the significance level associated with
# 	            CSSR's hypothesis tests.
# 	L        -- The maximum history length to look
#               back when inferring predictive
#               distributions.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Xt_name == '':
	axs = ['0']
	ays = ['0', '1']
else:
	axs = ['0', '1']
	ays = ['0', '1']

e_symbols = list(itertools.product(axs, ays)) # All of the possible pairs of emission
                                              # symbols for (x, y)

machine_fname = 'transCSSR_results/+{}.dot'.format(Xt_name)
transducer_fname = 'transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name)
inf_alg = 'transCSSR'

P, T_states_to_index, M_states_to_index, T_trans, M_trans = compute_mixed_transition_matrix(machine_fname, transducer_fname, axs, ays, inf_alg = inf_alg)

T_states = T_states_to_index.keys()
M_states = M_states_to_index.keys()

stationary_dist_mixed, stationary_dist_eT = compute_channel_states_distribution(P, M_states, T_states)

M_start_state = 'A'
T_start_state = 'A'

L_max = 12

N = L_max-1

num_sims = 5000

count_ones = numpy.zeros(L_max-1)

for sim_ind in range(num_sims):
	X = simulate_eM(N, 'transCSSR_results/+{}.dot'.format(Xt_name), ays, 'transCSSR', initial_state = M_start_state)

	Y = simulate_eT(N, 'transCSSR_results/+{}.dot'.format(Xt_name), 'transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name), X, axs, ays, 'transCSSR', initial_state = T_start_state)

	for y_ind, y in enumerate(Y):
		if y == '1':
			count_ones[y_ind] += 1

prop_ones = count_ones / num_sims

print prop_ones

for T_start_state in [T_start_state]:
# for T_start_state in T_states:
	print('\nStarting from transducer state {}...'.format(T_start_state))
	M_state_from = M_start_state
	T_state_from = T_start_state

	p_joint_string_Lp1 = numpy.zeros((len(axs), len(ays)))

	joint_string_prods = [{('', '') : 1.0}]
	joint_string_states = [{('', '') : (M_state_from, T_state_from)}]

	for L in range(L_max):
		joint_string_prods.append({})
		joint_string_states.append({})

		for xword, yword in joint_string_prods[-2].keys():
			for ax_ind, ax in enumerate(axs):
				x = ax
				for ay_ind, ay in enumerate(ays):
					y = ay

					p_prod = joint_string_prods[-2][(xword, yword)]

					if p_prod == 0.:
						pass
					else:
						M_state_from, T_state_from = joint_string_states[-2][(xword, yword)]

						T_state_to, pT_to = T_trans.get((T_state_from, x, y), (None, 0))

						if pT_to == 0:
							# break
							pass

						M_state_to, pM_to = M_trans.get((M_state_from, x), (None, 0))

						if pM_to == 0:
							# break
							pass

						joint_string_prods[-1][(xword + x, yword + y)] = p_prod*pT_to*pM_to
						joint_string_states[-1][(xword + x, yword + y)] = (M_state_to, T_state_to)
		# 		p_joint_string_Lp1[ax_ind, ay_ind] += p_eT*p_eM

		# print(p_joint_string_Lp1)

		# p_y_string = numpy.sum(p_joint_string_Lp1, 0)
		# p_y_string = p_y_string / numpy.sum(p_y_string)

		# print(p_y_string)

	# The total probability across the words at a given level should sum to 1.

	# for ind in range(len(joint_string_prods)):
	# 	tot_prob = 0.

	# 	for xword, yword in joint_string_prods[ind].keys():
	# 		tot_prob += joint_string_prods[ind][(xword, yword)]

	# 	print(ind, tot_prob)

	# Compute P(Y_{L} | S_{0} = s)

	pred_probs_by_L = numpy.zeros((L_max-1, len(ays)))

	for L in range(1,L_max):
		for xword, yword in joint_string_prods[L-1].keys():
			for x in axs:
				for ay_ind, y in enumerate(ays):
					p_prod = joint_string_prods[L].get((xword + x, yword + y), 0.0)

					pred_probs_by_L[L-1, ay_ind] += p_prod

		# print(pred_probs_by_L[L-1, :], pred_probs_by_L[L-1, :]/numpy.sum(pred_probs_by_L[L-1, :]))

		pred_probs_by_L[L-1, :] = pred_probs_by_L[L-1, :]/numpy.sum(pred_probs_by_L[L-1, :])

		print(pred_probs_by_L[L-1, :])

	import matplotlib.pyplot as plt

	plt.figure()
	plt.plot(pred_probs_by_L[:, 0])
	plt.plot(pred_probs_by_L[:, 1])
	plt.plot(prop_ones)
	plt.show()