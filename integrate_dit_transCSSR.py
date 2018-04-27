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

import dit

import time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# The various test transducers. Xt is the input
# and Yt is the output.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# data_prefix = 'NEURO-Set5/'
data_prefix = ''

# Yt_name = 'coinflip_through_even'
# Yt_name = 'coinflip_through_evenflip'
# Yt_name = 'coinflip_through_periodickick'
# Yt_name = 'coinflip_through_periodicevenkick'
# Yt_name = 'even_through_even'
# Yt_name = 'even'
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

# Yt_name = 'sample1'
# Yt_name = 'sample0'

# Xt_name = 'coinflip'
# Xt_name = 'even'
# Xt_name = ''
# Xt_name = 'barnettX'
# Xt_name = 'even-excite_w_refrac'
# Xt_name = 'tricoin'

# Xt_name = 'sample0'
# Xt_name = 'sample1'

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

Xt_name = 'Xt_delay-channel'
Yt_name = 'Yt_delay-channel'

# The odd random channel.

# Xt_name = 'Xt_odd-random-channel'
# Yt_name = 'Yt_odd-random-channel'

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

axs = ['0', '1']
ays = ['0', '1']

# axs = ['0']
# ays = ['0', '1']

e_symbols = list(itertools.product(axs, ays)) # All of the possible pairs of emission
                                              # symbols for (x, y)

alpha = 0.001

verbose = False

# L is the maximum amount we want to ever look back.

L_max = 11

Tx = len(stringX); Ty = len(stringY)

assert Tx == Ty, 'The two time series must have the same length.'

T = Tx

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Convert the joint X x Y alphabet to a joint
# alphabet J suitable for use with dit.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

word_count = 0

marg_alphas_to_joint_alpha = {}
joint_alpha_to_marg_alphas = {}

curr_alpha = 0
for ax in axs:
	for ay in ays:
		marg_alphas_to_joint_alpha[ax, ay] = curr_alpha
		joint_alpha_to_marg_alphas[curr_alpha] = (ax, ay)

		curr_alpha += 1

arrayJ = numpy.zeros(len(stringX), dtype = 'int16')

for t in range(len(stringX)):
	arrayJ[t] = marg_alphas_to_joint_alpha[stringX[t], stringY[t]]

startTime = time.time()
histories, cCounts, hCounts, alphabet = dit.inference.counts.counts_from_data(data = arrayJ, hLength = L_max+1, fLength = 1, marginals = True, standardize = True)
print ('The dit counting took {0} seconds...'.format(time.time() - startTime))

startTime = time.time()
word_lookup_fut = defaultdict(int)
word_lookup_marg = defaultdict(int)

for word_ind, word_dit in enumerate(histories):
	word_x = ''
	word_y = ''
	for w in word_dit:
		ax, ay = joint_alpha_to_marg_alphas[w]

		word_x += ax
		word_y += ay

	word_lookup_fut[word_x, word_y] = hCounts[word_ind]

for joint_words in word_lookup_fut.keys():
	if len(joint_words[0]) <= L_max:
		for ax in axs:
			c = 0

			for ay in ays:
				c += word_lookup_fut[joint_words[0] + ax, joint_words[1] + ay]

			word_lookup_marg[joint_words[0] + ax, joint_words[1]] = c
print ('Unpacking dit to transCSSR took {0} seconds...'.format(time.time() - startTime))

startTime = time.time()
word_lookup_marg_transCSSR, word_lookup_fut_transCSSR = estimate_predictive_distributions(stringX, stringY, L_max)
print ('The transCSSR counting took {0} seconds...'.format(time.time() - startTime))

# for word in word_lookup_fut_transCSSR.keys()[:50]:
# 	print(word, word_lookup_fut_transCSSR[word], word_lookup_fut[word])

# for word in word_lookup_marg_transCSSR.keys()[:50]:
# 	print(word, word_lookup_marg_transCSSR[word], word_lookup_marg[word])

# print("Running transCSSR...")
# epsilon, invepsilon, morph_by_state = run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, alpha = alpha, verbose = False)

# print 'The epsilon-transducer has {} states.'.format(len(invepsilon))

# print_morph_by_states(morph_by_state, axs, ays, e_symbols)

# filtered_states, filtered_probs, stringY_pred = filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L_max)

# print 'Xt Yt \hat\{Y\}t St P(Yt = 1 | Xt, St)'

# for t_ind in range(int(numpy.min([100, len(stringX)]))):
# 	print stringX[t_ind], stringY[t_ind], stringY_pred[t_ind], filtered_states[t_ind], filtered_probs[t_ind]