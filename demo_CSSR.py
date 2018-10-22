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

# Yt is the output. Xt should be set to the null string.

data_prefix = ''

# Yt_name = 'coinflip_through_even'
# Yt_name = 'coinflip_through_evenflip'
# Yt_name = 'coinflip_through_periodickick'
# Yt_name = 'coinflip_through_periodicevenkick'
# Yt_name = 'even_through_even'
Yt_name = 'even'
# Yt_name = 'rip'
# Yt_name = 'rip-rev'
# Yt_name = 'barnettY'
# Yt_name = 'even-excite_w_refrac'
# Yt_name = 'coinflip-excite_w_refrac'
# Yt_name = 'coinflip'
# Yt_name = 'period4'
# Yt_name = 'golden-mean'
# Yt_name = 'golden-mean-rev'
# Yt_name = 'complex-csm'
# Yt_name = 'tricoin_through_singh-machine'
# Yt_name = 'coinflip_through_floatreset'
# Yt_name = '1mm_sim'

Xt_name = ''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Load in the data for each process.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

stringY = open('data/{}{}.dat'.format(data_prefix, Yt_name)).readline().strip()

stringY = stringY[:10000]

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

axs = ['0']
ays = ['0', '1']

e_symbols = list(itertools.product(axs, ays)) # All of the possible pairs of emission
                                              # symbols for (x, y)

alpha = 0.001

verbose = False

# L is the maximum amount we want to ever look back.

L_max = 9

inf_alg = 'transCSSR'

Tx = len(stringX); Ty = len(stringY)

assert Tx == Ty, 'The two time series must have the same length.'

T = Tx

word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(stringX, stringY, L_max)

epsilon, invepsilon, morph_by_state = run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, alpha = alpha)

print 'The epsilon-transducer has {} states.'.format(len(invepsilon))

print_morph_by_states(morph_by_state, axs, ays, e_symbols)

filtered_states, filtered_probs, stringY_pred = filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L_max)

machine_fname = 'transCSSR_results/+.dot'
transducer_fname = 'transCSSR_results/+{}.dot'.format(Yt_name)

pred_probs_by_time, cur_states_by_time = filter_and_pred_probs(stringX, stringY, machine_fname, transducer_fname, axs, ays, inf_alg, verbose_filtering_errors = True)

pred_probs_by_time_break, cur_states_by_time_break = filter_and_pred_probs_breakforbidden(stringX, stringY, machine_fname, transducer_fname, axs, ays, inf_alg)

for t in range(30):
	print(t, stringY[t], filtered_probs[t], pred_probs_by_time_break[t, 1], pred_probs_by_time[t, 1])

import matplotlib.pyplot as plt
plt.ion()

plt.figure()
plt.plot(filtered_probs, label = 'filtered_probs')
plt.plot(pred_probs_by_time_break[:, 1], label = 'pred_probs_by_time_break')
plt.plot(pred_probs_by_time[:, 1], label = 'pred_probs_by_time')
plt.xlim([0, 20])
plt.legend()