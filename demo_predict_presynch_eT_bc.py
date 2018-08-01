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
# Load in the data for each process.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stringY = open('data/{}{}.dat'.format(data_prefix, Yt_name)).readline().strip()

if Xt_name == '':
	stringX = '0'*len(stringY)
else:
	stringX = open('data/{}{}.dat'.format(data_prefix, Xt_name)).readline().strip()

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

alpha = 0.001

verbose = False

# L is the maximum amount we want to ever look back.

L_max = 3

Tx = len(stringX); Ty = len(stringY)

assert Tx == Ty, 'The two time series must have the same length.'

T = Tx

word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(stringX, stringY, L_max)

epsilon, invepsilon, morph_by_state = run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, alpha = alpha, verbose = False)

ind_go_to = 20

possible_states_from_predict_presynch_eT = numpy.zeros((ind_go_to-1, len(invepsilon)), dtype = numpy.int32)

for cur_ind in range(1, ind_go_to):
	curX = stringX[:cur_ind]
	curY = stringY[:cur_ind-1]

	preds, possible_states = predict_presynch_eT(curX, curY, machine_fname = 'transCSSR_results/+{}.dot'.format(Xt_name), transducer_fname = 'transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name), axs = axs, ays = ays, inf_alg = 'transCSSR')

	possible_states_from_predict_presynch_eT[cur_ind - 1] = possible_states

	print(cur_ind, curX, curY + '*', preds.tolist(), possible_states)

print('')

preds_all, possible_states_all = filter_and_pred_probs(stringX, stringY, machine_fname = 'transCSSR_results/+{}.dot'.format(Xt_name), transducer_fname = 'transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name), axs = axs, ays = ays, inf_alg = 'transCSSR')

for cur_ind in range(1, ind_go_to):
	curX = stringX[:cur_ind]
	curY = stringY[:cur_ind-1]

	print(cur_ind, curX, curY + '*', preds_all[cur_ind-1, :].tolist(), possible_states_all[cur_ind-1, :].tolist())

filtered_states, filtered_probs, stringY_pred = filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L_max, memoryless = False)

print_go_to = 40

print("\n\nFirst {} predictions.".format(print_go_to))
for ind in range(print_go_to):
	print(filtered_probs[ind], preds_all[ind, 1])

print("\n\nLast {} predictions.".format(print_go_to))
for ind in range(preds_all.shape[0] - print_go_to, preds_all.shape[0]):
	print(filtered_probs[ind], preds_all[ind, 1])

import matplotlib.pyplot as plt

plt.figure()
plt.plot(filtered_probs, label = 'Using filter_and_predict')
plt.plot(preds_all[:, 1], label = 'Using filter_and_pred_probs')
plt.xlim([0, 1000])
plt.legend()
plt.show()