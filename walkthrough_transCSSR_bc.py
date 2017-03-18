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

def print_stochastic_matrix(past, axs, ays):
	xpast = past[0]; ypast = past[1]
	
	for ax in axs:
		cur_row = ''
		for ay in ays:
			if word_lookup_marg[(xpast + ax, ypast)] == 0:
				cur_row += '\t' + 'na'
			else:
				cur_row += '\t' + str(word_lookup_fut[(xpast + ax, ypast + ay)]/float(word_lookup_marg[(xpast + ax, ypast)]))
		
		print cur_row + '\n'
	
	
	print '\n\n'
	
	for ax in axs:
		cur_row = ''
		for ay in ays:
			cur_row += '\t' + str(word_lookup_fut[(xpast + ax, ypast + ay)])
		
		print cur_row + '\n'

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

# Xt_name = 'Xt_delay-channel'
# Yt_name = 'Yt_delay-channel'

# The odd random channel.

Xt_name = 'Xt_odd-random-channel'
Yt_name = 'Yt_odd-random-channel'

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

# axs = ['0', '1', '2']
# ays = ['0', '1']

e_symbols = list(itertools.product(axs, ays)) # All of the possible pairs of emission
                                              # symbols for (x, y)

# The number of degrees of freedom associated with 

alpha = 0.001

verbose = False

# L is the maximum amount we want to ever look back.

L_max = 3

Tx = len(stringX); Ty = len(stringY)

assert Tx == Ty, 'The two time series must have the same length.'

T = Tx

word_lookup_marg, word_lookup_fut = estimate_predictive_distributions(stringX, stringY, L_max)

epsilon, invepsilon, morph_by_state = run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, alpha = alpha, verbose = True)

print 'The epsilon-transducer has {} states.'.format(len(invepsilon))

# os.system('open transCSSR_results/mydot-det_transients.dot'.format(Xt_name, Yt_name))
# os.system('open transCSSR_results/mydot-det_recurrent.dot'.format(Xt_name, Yt_name))

if len(invepsilon) < 30:
	os.system('open transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name))
else:
	print 'Warning: The epsilon-transducer has {} states, so we will not open the dot file.'

os.system('open transCSSR_results/{}+{}.dat_results'.format(Xt_name, Yt_name))

print_morph_by_states(morph_by_state, axs, ays, e_symbols)