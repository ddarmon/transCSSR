import numpy
import scipy.stats
import itertools
import copy
import string
import os

from collections import Counter, defaultdict
from filter_data_methods import *
from igraph import *

from transCSSR import *



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# The various test transducers. Xt is the input
# and Yt is the output.
#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data_prefix = ''

Xt_name = ''

y_name_a = 'even'
# y_name_a = 'even-perturbed'
# y_name_a = 'golden-mean'
# y_name_a = 'coinflip'
# y_name_b = 'even'
# y_name_b = 'even-perturbed'
y_name_b = 'golden-mean'
# y_name_b = 'golden-mean-perturbed'
# y_name_b = 'coinflip'

axs = ['0', '1']
ays = ['0', '1']

L_word = 7

# for L_word in range(1, 20):
for L_word in [L_word]:
	L_pow = 2**L_word

	kl_div = 0
	l1_dist = 0

	for i_word in range(L_pow):
		xs = format(i_word, '0{}b'.format(L_word))
	
		p_word_a = compute_word_probability_eM(xs, 'transCSSR_results/{}+{}.dot'.format(Xt_name, y_name_a), axs, 'transCSSR')
		p_word_b = compute_word_probability_eM(xs, 'transCSSR_results/{}+{}.dot'.format(Xt_name, y_name_b), axs, 'transCSSR')
	
		print xs, p_word_a, p_word_b
	
		if p_word_a == 0 and p_word_b == 0:
			pass
		elif p_word_a == 0 and p_word_b != 0:
			pass
		elif p_word_a != 0 and p_word_b == 0:
			kl_div = numpy.inf
		else:
			kl_div += p_word_a*numpy.log2(p_word_a/p_word_b)
		
		l1_dist += numpy.abs(p_word_a - p_word_b)

	print 'The KL-divergence is {}\nThe L1-distance is {} \n\n'.format(kl_div/L_word, 0.5*l1_dist)

P, M_states_to_index, M_trans = compute_eM_transition_matrix('transCSSR_results/{}+{}.dot'.format(Xt_name, y_name_a), axs, 'transCSSR')
P_uniform, M_states_to_index, M_trans_uniform = compute_eM_transition_matrix_uniform('transCSSR_results/{}+{}.dot'.format(Xt_name, y_name_a), axs, 'transCSSR')

stationary_dist_mixed, stationary_dist_eM = compute_channel_states_distribution(P, {'A' : 0}, M_states_to_index)

for L_word in [L_word]:
	L_pow = 2**L_word

	kl_div = 0
	l1_dist = 0

	for i_word in range(L_pow):
		xs = format(i_word, '0{}b'.format(L_word))
	
		p_word_a = compute_word_probability_eM(xs, 'transCSSR_results/{}+{}.dot'.format(Xt_name, y_name_a), axs, 'transCSSR', uniform = True)
		p_word_b = compute_word_probability_eM(xs, 'transCSSR_results/{}+{}.dot'.format(Xt_name, y_name_b), axs, 'transCSSR', uniform = True)
	
		print xs, p_word_a, p_word_b
	
		if p_word_a == 0 and p_word_b == 0:
			pass
		elif p_word_a == 0 and p_word_b != 0:
			pass
		elif p_word_a != 0 and p_word_b == 0:
			kl_div = numpy.inf
		else:
			kl_div += p_word_a*numpy.log2(p_word_a/p_word_b)
			
		l1_dist += numpy.abs(p_word_a - p_word_b)

	print 'The KL-divergence is {}'.format(kl_div/L_word)
	print 'The L1-distance is {}'.format(0.5*l1_dist)