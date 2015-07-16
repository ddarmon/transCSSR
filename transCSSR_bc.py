from collections import Counter, defaultdict
import random
import os
import itertools
import string
import copy
import pylab
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# Dependencies: numpy, scipy, pandas, igraph, pylab, matplotlib

import numpy
import scipy.stats
import pandas
from igraph import *

from filter_data_methods import *

def chisquared_test(morph1, morph2, axs, ays, alpha = 0.001, test_type = 'chi2'):
	"""
	Compare two predictive distributions (morph1 and morph2) to determine
	if they are (statistically) equivalent. This can be done using 
	one of the chi-squared test or the G-test.

	Parameters
	----------
	morph1 : list
			the counts associated with the first predictive distribution
	morph2 : list
			the counts associated with the second predictive distribution
	df : int
			The degrees of freedom associated with the hypothesis test,
			equal to len(morph*) - 1.
	alpha : float
			The significance level for which we reject the null hypothesis
			that the two morphs result from the same distribution.
	test_type : str
			The statistic used in the hypothesis test, one of 'chi2'
			(for the chi-squared statistic) or 'G' (for the 
			log-likelihood ratio / G statistic).

	Returns
	-------
	test : bool
			If test is true, we reject the null hypothesis that
			the two predictive distributions are the same, at level
			alpha.
	pvalue : float
			The p-value associated with the hypothesis test.
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	if test_type == 'chi2':
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#
		# New way: chi-squared, direct.
		#
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		# NOTE: We are really comparing stochastic matrices,
		# P(Y_{t} = y | X_{t} = x, Past = past), 
		# because of the conditioning on x. Thus, we compute
		# the chi-squared statistics 'row-wise' (i.e. for
		# fixed x), and then sum them to get the total
		# chi-squared statistic in the end.
		
		# morph* stores the counts
		# 	# {Xt = x, Yt = y, Past = past}
		# stacked with all of the x's together.
		
		chisquared_statistic = 0
		
		df_x = 0
		
		array_morph1 = numpy.array(morph1); array_morph2 = numpy.array(morph2)
		
		for row_ind in range(len(axs)):
			tmp_morph1 = array_morph1[row_ind*len(axs):row_ind*len(axs) + len(ays)]
			tmp_morph2 = array_morph2[row_ind*len(axs):row_ind*len(axs) + len(ays)]
			
			if numpy.sum(tmp_morph1) == 0 or numpy.sum(tmp_morph2) == 0: # If we haven't seen a (X_{t}, past), don't hold that against the causal state.
				pass
			else:
				df_x += 1 # This row is non-zero in the stochastic matrix.
				n1 = numpy.sum(tmp_morph1)
				n2 = numpy.sum(tmp_morph2)

				ns = [n1, n2]

				theta0 = (tmp_morph1 + tmp_morph2)/float(n1 + n2)

				contingency_table = numpy.vstack((tmp_morph1, tmp_morph2))

				for array_ind in range(len(tmp_morph1)):
					for sample_ind in range(2):
						denominator = ns[sample_ind]*theta0[array_ind]
						if denominator == 0:
							pass
						else:
							numerator = numpy.power(contingency_table[sample_ind, array_ind] - ns[sample_ind]*theta0[array_ind], 2)

							chisquared_statistic += numerator / float(denominator)

			# The quantile (inverse CDF) for a standard chi-square variable
		
		df = df_x*(len(ays)-1)
		
		quantile = scipy.stats.chi2.ppf
		F = scipy.stats.chi2.cdf

		test = (chisquared_statistic > quantile(1-alpha, df)) # If true, we reject the null hypothesis. Otherwise, we do not.

		pvalue = 1 - F(chisquared_statistic, df)
	elif test_type == 'G':
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		#
		# New way: G-statistic.
		#
		#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
		chisquared_statistic = 0
		
		df_x = 0
		
		array_morph1 = numpy.array(morph1); array_morph2 = numpy.array(morph2)
		
		for row_ind in range(len(axs)):
			tmp_morph1 = array_morph1[row_ind*len(axs):row_ind*len(axs) + len(ays)]
			tmp_morph2 = array_morph2[row_ind*len(axs):row_ind*len(axs) + len(ays)]
			
			if numpy.sum(tmp_morph1) == 0 or numpy.sum(tmp_morph2) == 0:
				pass
			else:
				df_x += 1 # This row is non-zero in the stochastic matrix.
				n1 = numpy.sum(tmp_morph1)
				n2 = numpy.sum(tmp_morph2)

				ns = [n1, n2]

				theta0 = (numpy.array(tmp_morph1) + numpy.array(tmp_morph2))/float(n1 + n2)

				contingency_table = numpy.vstack((tmp_morph1, tmp_morph2))

				for array_ind in range(len(tmp_morph1)):
					for sample_ind in range(2):
						denominator = ns[sample_ind]*theta0[array_ind]
						if denominator == 0:
							pass
						else:
							numerator = contingency_table[sample_ind, array_ind]

							if numerator == 0:
								pass
							else:
								chisquared_statistic += numerator*numpy.log(numerator/float(denominator))

		chisquared_statistic = 2*chisquared_statistic
		
		df = df_x*(len(ays) - 1)
		
		# The quantile (inverse CDF) for a standard chi-square variable

		quantile = scipy.stats.chi2.ppf
		F = scipy.stats.chi2.cdf

		test = (chisquared_statistic > quantile(1-alpha, df)) # If true, we reject the null hypothesis. Otherwise, we do not.

		pvalue = 1 - F(chisquared_statistic, df)
	
	return test, pvalue

def get_connected_component(epsilon, invepsilon, e_symbols, L_max):
	"""
	For a finite-state Markov chain, find the largest
	component of the graph associated with the state
	transition matrix.
	
	This version works with *memoryful* transducers,
	where we assume that the output at time t depends
	on the joint (input, output) at previous times.

	Parameters
	----------
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	e_symbols : list
			The emission symbols associated with (X, Y).
	L_max : int
			The maximum history length used in estimating the
			predictive distributions.

	Returns
	-------
	clusters : list
			The connected components for the state-transition graph.
	state_matrix : matrix
			A binary matrix where state_matrix[i, j] == 1 indicates 
			that transitions are allowed from state i to state j.
	trans_dict : dict
			A dictionary mapping a state to the allowed transitions
			from that state, i.e.
				trans_dict = {from_state : [to_state1, to_state2, ...], ...}
	states_to_index : dict
			A dictionary mapping from the states to their index in 
			state_matrix.
	index_to_states : dict
			A dictionary mapping from the indices in state_matrix to
			the original state labels
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	states = invepsilon.keys()

	state_matrix = numpy.zeros((len(states), len(states)))
	
	# Also store the allowed transitions as an edge list,
	# for easier checking. Use a dictionary of the form
	# {from_state : [to_state1, to_state2, ...]}.
	
	trans_dict = defaultdict(dict)

	states_to_index = {}
	index_to_states = {}

	for state_index, state in enumerate(states):
		states_to_index[state] = state_index
		index_to_states[state_index] = state

	for state in states:
		i = states_to_index[state]
		
		need_Lmax = True # Whether or not we need to use the length L_max histories in
						 # defining the transition structure.
		
		for hist in invepsilon[state].keys():
			if len(hist[0]) == L_max - 1:
				need_Lmax = False
		
		for hist in invepsilon[state].keys():
			if len(hist[0]) == L_max:
				if need_Lmax:
					for e_symbol in e_symbols:
						s = epsilon.get((hist[0][1:] + e_symbol[0], hist[1][1:] + e_symbol[1]), -1)
				
						if s != -1:
							j = states_to_index[s]
					
							state_matrix[i, j] = 1
						
							trans_dict[i][j] = True
				else:
					pass
			else:
				for e_symbol in e_symbols:
					s = epsilon.get((hist[0] + e_symbol[0], hist[1] + e_symbol[1]), -1)
			
					if s != -1:
						j = states_to_index[s]
				
						state_matrix[i, j] = 1
					
						trans_dict[i][j] = True
		

	g = Graph.Adjacency(state_matrix.tolist())

	clusters = g.clusters()

	return clusters, state_matrix, trans_dict, states_to_index, index_to_states
def draw_dot(fname, epsilon, invepsilon, morph_by_state, axs, ays, L_max):
	"""
	This function draws the .dot file associated with the 
	epsilon-transducer stored in epsilon+invepsilon.
	
	This version works with *memoryful* transducers,
	where we assume that the output at time t depends
	on the joint (input, output) at previous times.

	Parameters
	----------
	fname : str
			The name for the .dot file.
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	axs : list
			The emission symbols associated with X.
	ays : list
			The emission symbols associated with Y.
	L_max : int
			The maximum history length used in estimating the
			predictive distributions.
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	input_lookup = {} # Create an ordering of the input alphabet $\mathcal{X}$,
					  # for indexing into the predictive distribution.
	
	for x_ind, x in enumerate(axs):
		input_lookup[x] = x_ind
	
	output_lookup = {} # Create an ordering of the output alphabet $\mathcal{Y}$,
					   # for indexing into the predictive distribution.
	
	for y_ind, y in enumerate(ays):
		output_lookup[y] = y_ind
	
	# Recall that, for a fixed causal state s, prob_by_state is a 
	# row-stochastic matrix. However, we store that matrix as an
	# array, where we stack *rows*. So entry W[x, y] is 
	# stored as prob_by_state[len(ays)*x + y].
	
	prob_by_state = {}
	
	for state in invepsilon:
		prob_by_state[state] = numpy.zeros(len(axs)*len(ays))
	
		for x_ind in range(len(axs)):
			counts_by_x = numpy.array(morph_by_state[state], dtype = 'float')[x_ind*len(ays):x_ind*len(ays) + len(ays)]
			prob_by_x = counts_by_x/numpy.sum(counts_by_x)
		
			prob_by_state[state][x_ind*len(ays):x_ind*len(ays) + len(ays)] = prob_by_x
	
	dot_header = 'digraph  {\nsize = \"6,8.5\";\nratio = "fill";\nnode\n[shape = circle];\nnode [fontsize = 24];\nnode [penwidth = 5];\nedge [fontsize = 24];\nnode [fontname = \"CMU Serif Roman\"];\ngraph [fontname = \"CMU Serif Roman\"];\nedge [fontname = \"CMU Serif Roman\"];\n'

	with open('{}.dot'.format(fname), 'w') as wfile:
		wfile.write(dot_header)

		# Draw associated candidate CSM.
		
		# Report the states as 0 through (# states) - 1.
		
		printing_lookup = {}
		
		for state_rank, state in enumerate(invepsilon.keys()):
			printing_lookup[state] = state_rank
		
		seen_transition = {}
		
		for state in invepsilon.keys():
			need_Lmax = True # Whether or not we need to use the length L_max histories in
							 # defining the transition structure.
			
			for hist in invepsilon[state].keys():
				if len(hist[0]) == L_max - 1:
					need_Lmax = False
			
			for history in invepsilon[state]:
				if len(history[0]) == L_max:
					if need_Lmax:
						for ay in ays:
							for ax in axs:
								to_state = epsilon.get((history[0][1:] + ax, history[1][1:] + ay), -1)

								if to_state == -1:
									pass
								else:
									if seen_transition.get((state, to_state, (ax, ay)), False):
										pass
									else:
										seen_transition[(state, to_state, (ax, ay))] = True
							
										wfile.write('{} -> {} [label = \"{}|{}:{:.3}\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ay, ax, prob_by_state[state][len(ays)*input_lookup[ax] + output_lookup[ay]]))
					else:
						pass
				else:
					for ay in ays:
						for ax in axs:
							to_state = epsilon.get((history[0] + ax, history[1] + ay), -1)

							if to_state == -1:
								pass
							else:
								if seen_transition.get((state, to_state, (ax, ay)), False):
									pass
								else:
									seen_transition[(state, to_state, (ax, ay))] = True
							
									wfile.write('{} -> {} [label = \"{}|{}:{:.3}\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ay, ax, prob_by_state[state][len(ays)*input_lookup[ax] + output_lookup[ay]]))
		wfile.write('}')
def draw_dot_singlearrows(fname, epsilon, invepsilon, morph_by_state, axs, ays, L_max):
	"""
	This function draws the .dot file associated with the 
	epsilon-transducer stored in epsilon+invepsilon.
	
	This version works with *memoryful* transducers,
	where we assume that the output at time t depends
	on the joint (input, output) at previous times.

	Parameters
	----------
	fname : str
			The name for the .dot file.
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	axs : list
			The emission symbols associated with X.
	ays : list
			The emission symbols associated with Y.
	L_max : int
			The maximum history length used in estimating the
			predictive distributions.
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	input_lookup = {} # Create an ordering of the input alphabet $\mathcal{X}$,
					  # for indexing into the predictive distribution.
	
	for x_ind, x in enumerate(axs):
		input_lookup[x] = x_ind
	
	output_lookup = {} # Create an ordering of the output alphabet $\mathcal{Y}$,
					   # for indexing into the predictive distribution.
	
	for y_ind, y in enumerate(ays):
		output_lookup[y] = y_ind
	
	# Recall that, for a fixed causal state s, prob_by_state is a 
	# row-stochastic matrix. However, we store that matrix as an
	# array, where we stack *rows*. So entry W[x, y] is 
	# stored as prob_by_state[len(ays)*x + y].
	
	prob_by_state = {}
	
	for state in invepsilon:
		prob_by_state[state] = numpy.zeros(len(axs)*len(ays))
	
		for x_ind in range(len(axs)):
			counts_by_x = numpy.array(morph_by_state[state], dtype = 'float')[x_ind*len(ays):x_ind*len(ays) + len(ays)]
			prob_by_x = counts_by_x/numpy.sum(counts_by_x)
		
			prob_by_state[state][x_ind*len(ays):x_ind*len(ays) + len(ays)] = prob_by_x
	
	dot_header = 'digraph  {\nsize = \"6,8.5\";\nratio = "fill";\nnode\n[shape = circle];\nnode [fontsize = 24];\nnode [penwidth = 5];\nedge [fontsize = 24];\nnode [fontname = \"CMU Serif Roman\"];\ngraph [fontname = \"CMU Serif Roman\"];\nedge [fontname = \"CMU Serif Roman\"];\n'

	with open('{}.dot'.format(fname), 'w') as wfile:
		wfile.write(dot_header)

		# Draw associated candidate CSM.
		
		# Report the states as 0 through (# states) - 1.
		
		printing_lookup = {}
		
		for state_rank, state in enumerate(invepsilon.keys()):
			printing_lookup[state] = state_rank
		
		seen_transition = {}
		
		exists_transition = {} # Whether a transition exists (from_state, to_state)
		
		W = defaultdict(str) # The stochastic matrix, stored as a string, by state
		
		for state in invepsilon.keys():
			need_Lmax = True # Whether or not we need to use the length L_max histories in
							 # defining the transition structure.
			
			for hist in invepsilon[state].keys():
				if len(hist[0]) == L_max - 1:
					need_Lmax = False
			
			for history in invepsilon[state]:
				if len(history[0]) == L_max:
					if need_Lmax:
						for ay in ays:
							for ax in axs:
								to_state = epsilon.get((history[0][1:] + ax, history[1][1:] + ay), -1)

								if to_state == -1:
									pass
								else:
									if seen_transition.get((state, to_state, (ax, ay)), False):
										pass
									else:
										seen_transition[(state, to_state, (ax, ay))] = True
										
										exists_transition[(state, to_state)] = True
										
										W[(state, to_state)] += '{}|{}:{:.3}\\l'.format(ay, ax, prob_by_state[state][len(ays)*input_lookup[ax] + output_lookup[ay]])
							
										# wfile.write('{} -> {} [label = \"({}, {})\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ax, ay))
					else:
						pass
				else:
					for ay in ays:
						for ax in axs:
							to_state = epsilon.get((history[0] + ax, history[1] + ay), -1)

							if to_state == -1:
								pass
							else:
								if seen_transition.get((state, to_state, (ax, ay)), False):
									pass
								else:
									seen_transition[(state, to_state, (ax, ay))] = True
							
									exists_transition[(state, to_state)] = True
									
									W[(state, to_state)] += '{}|{}:{:.3}\\l'.format(ay, ax, prob_by_state[state][len(ays)*input_lookup[ax] + output_lookup[ay]])
		
		for from_state in invepsilon.keys():
			for to_state in invepsilon.keys():
				if exists_transition.get((from_state, to_state), False):
					wfile.write('{} -> {} [label = \"{}\"];\n'.format(numeric_to_alpha(printing_lookup[from_state]), numeric_to_alpha(printing_lookup[to_state]), W[(from_state, to_state)]))
		
		wfile.write('}')
def numeric_to_alpha(value):
	"""
	This function maps from numeric values in {0, 1, 2, ...}
	to the Roman alphabet, {A, B, ..., Z, AA, BB, ...}.
	
	This function is for labeling the .dot file associated
	with an epsilon-transducer.

	Parameters
	----------
	value : int
			A numeric value in {0, 1, 2, ...}

	Returns
	-------
	output : str
			The Roman alphabet associated with
			value.
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	# Take a numeric value from {0, 1, ...} and
	# map it to {A, B, ..., Z, AA, ...}.
	
	remainder = value%26
	integer_part = value/26
	
	return (integer_part + 1)*string.uppercase[remainder]

def save_states(fname, epsilon, invepsilon, morph_by_state, axs, ays, L_max):
	"""
	Save the states associated with the epsilon-transducer to 
	a .dat_results file. The format of the .dat_results file
	follows the conventions from the CSSR code.
	
	This version works with *memoryful* transducers,
	where we assume that the output at time t depends
	on the joint (input, output) at previous times.

	Parameters
	----------
	fname : str
			The prefix of the .dat_results file.
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	morph_by_state : list
			The counts associated with the predictive distribution
			for a particular state.
	axs : list
			The emission symbols associated with X.
	ays : list
			The emission symbols associated with Y.
	L_max : int
			The maximum history length used in estimating the
			predictive distributions.

	Returns
	-------
	None
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	e_symbols = list(itertools.product(axs, ays))
	
	with open('{}.dat_results'.format(fname), 'w') as wfile:
		# Report the states as 0 through (# states) - 1.
	
		printing_lookup = {}
	
		for state_rank, state in enumerate(invepsilon.keys()):
			printing_lookup[state] = state_rank
		
		printing_lookup['NULL'] = 'NULL'
		
		for state_rank, state in enumerate(invepsilon):
			wfile.write('State number: {}\n'.format(state_rank))
			histories = invepsilon[state]
			
			# Report the histories in lexicographical order,
			# starting with all of the states of length L_max - 1
			# and then L_max.
			
			hists_short = []
			hists_long  = []

			for hist in histories:
				if len(hist[0]) == L_max:
					hists_long.append(hist)
				else:
					hists_short.append(hist)
			
			hists_short.sort(); hists_long.sort() # Sort in lexicographical order
			
			for history in hists_short:
				wfile.write('{}, {}\n'.format(history[0], history[1]))
			
			for history in hists_long:
				wfile.write('{}, {}\n'.format(history[0], history[1]))
			
			to_print = 'distribution: \n'
			
			# Note: This is really a *condensed* version of a
			# len(axs) by len(ays) stochastic matrix giving
			#		P(Yt | Xt, state)
			# into a single len(axs)*len(ays) array, for convenience.
			
			prob_by_state = numpy.zeros(len(axs)*len(ays))
			
			for x_ind in range(len(axs)):
				counts_by_x = numpy.array(morph_by_state[state], dtype = 'float')[x_ind*len(ays):x_ind*len(ays) + len(ays)]
				prob_by_x = counts_by_x/numpy.sum(counts_by_x)
				
				prob_by_state[x_ind*len(ays):x_ind*len(ays) + len(ays)] = prob_by_x
			
			for emission_ind, e_symbol in enumerate(e_symbols):
				to_print += 'P({}|{},state) = {}\t'.format(e_symbol[1], e_symbol[0], prob_by_state[emission_ind])
			
			# for x_ind, x_sym in enumerate(axs):
			# 	for y_ind, y_sym in enumerate(ays):
			# 		to_print += 'P({}|{},state) = {}\t'.format(x_sym, y_sym, prob_by_state[x_ind*len(axs) + y_ind])
				
				to_print += '\n'
				
			wfile.write(to_print)
			
			to_print = 'transitions: '
			
			for e_symbol in e_symbols:
				sample_history = invepsilon[state].keys()[0]
				
				if len(sample_history[0]) == L_max - 1:
					to_state = epsilon.get((sample_history[0] + e_symbol[0], sample_history[1] + e_symbol[1]), 'NULL')
				else:
					to_state = epsilon.get((sample_history[0][1:] + e_symbol[0], sample_history[1][1:] + e_symbol[1]), 'NULL')
				
				to_print += 'T(({}, {})) = {}\t'.format(e_symbol[0], e_symbol[1], printing_lookup[to_state])
			
			wfile.write(to_print + '\n')
			
			wfile.write('P(State) = ...')
			
			wfile.write('\n\n')
def print_transitions(epsilon, invepsilon):
	"""
	For a given partition of histories of length
	L_max or smaller, print the transitions
	from state to state based on appending
	a new (input, output) emission symbol to 
	a given history.
	
	This function can therefore be used to check
	determinism / unifilarity 'by hand.'

	Parameters
	----------
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.

	Returns
	-------
	None
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	for state in invepsilon.keys():
		print 'On precausal state {}'.format(state) + '\n' + 10*'='
		for history in invepsilon[state]:
			if len(history[0]) == L_max-1:
				for ay in ays:
					for ax in axs:
						to_state = epsilon.get((history[0] + ax, history[1] + ay), -1)

						if to_state == -1:
							pass
						else:
							print 'History {} transitions to state {} on emission {}'.format(history, to_state, (ax, ay))
	

def remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max, memoryless = False, verbose = False):
	"""
	This function removes states that have become transient
	after applying the determinization procedure to the
	candidate causal states.
	
	This function uses remove_states to change the
	structures epsilon, invepsilon, and morph_by_state
	in place, and therefore does not return anything.

	Parameters
	----------
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	morph_by_state : list
			The counts associated with the predictive distribution
			for a particular state.
	e_symbols : list
			The emission symbols associated with (X, Y).
	L_max : int
			The maximum history length used in estimating the
			predictive distributions.
	memoryless : bool
			True if we consider an (output) memoryless transducer,
			False otherwise.
	verbose : bool
			True if we wish to print out various intermediary
			values during the removal of the transient states.

	Returns
	-------
	None
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	# Remove transient states.
	
	if memoryless:
		clusters, state_matrix, trans_dict, states_to_index, index_to_states = get_connected_component_memoryless(epsilon, invepsilon, e_symbols, L_max)
	else:
		clusters, state_matrix, trans_dict, states_to_index, index_to_states = get_connected_component(epsilon, invepsilon, e_symbols, L_max)
	
	if verbose:
		print clusters.membership
	
	remove_states = {}
	
	for from_state_ind in trans_dict:
		for to_state_ind in trans_dict[from_state_ind]:
			if clusters.membership[from_state_ind] != clusters.membership[to_state_ind]:
				remove_states[index_to_states[from_state_ind]] = True
	
	for s in remove_states:
		remove_state(s, epsilon, invepsilon, morph_by_state)
	
	# If more than one recurrent strongly connected componenet
	# remains, only keep the largest one.
	
	if memoryless:
		clusters, state_matrix, trans_dict, states_to_index, index_to_states = get_connected_component_memoryless(epsilon, invepsilon, e_symbols, L_max)
	else:
		clusters, state_matrix, trans_dict, states_to_index, index_to_states = get_connected_component(epsilon, invepsilon, e_symbols, L_max)
	
	if verbose:
		print clusters.membership
	
	# Mode returns two values: the mode, and the count associated with the mode.
	
	keep_component, count = map(int, scipy.stats.mode(clusters.membership))
	
	for state_ind, membership in enumerate(clusters.membership):
		if membership == keep_component:
			pass
		else:
			s = index_to_states[state_ind]

			remove_state(s, epsilon, invepsilon, morph_by_state)

def remove_state(state, epsilon, invepsilon, morph_by_state):
	"""
	Removes the passed state from invepsilon, the histories
	associated with that state from epsilon, the morph associated
	with that state from morph_by_state.

	Parameters
	----------
	state : int
			The numeric value associated with the causal state
			to be removed.
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	morph_by_state : list
			The counts associated with the predictive distribution
			for a particular state.

	Returns
	-------
	None

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	for hist in invepsilon[state]:
		epsilon.pop(hist)
		
	invepsilon.pop(state)
	
	morph_by_state.pop(state)

def print_morph_by_states(morph_by_state, axs, ays, e_symbols):
	"""
	Print the probabilities associated with the 
	counts in morph_by_state. Thus, this prints
		P(Y_{t} | S_{t-1} = s_{t-1})
	The predictive distribution for the output Y_{t}
	conditional on the causal state s_{t-1}.

	Parameters
	----------
	morph_by_state : list
			The counts associated with the predictive distribution
			for a particular state.

	Returns
	-------
	None

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	for state in morph_by_state:
		for x_ind in range(len(axs)):
			morph = numpy.array(morph_by_state[state])[x_ind*len(ays):x_ind*len(ays) + len(ays)]
		
			tot = float(numpy.sum(morph))
		
			print 'state = {}, x = {}, P(Yt | Xt, state) = {}'.format(state, axs[x_ind], numpy.divide(morph, tot))


def print_counts(hist, word_lookup_marg, word_lookup_fut, e_symbols):
	"""
	Print the counts and the estimated predictive distribution 
	associated with the history hist = (xhist, yhist).

	Parameters
	----------
	hist : tuple
			A tuple of the form (xhist, yhist) where
			xhist and yhist are strings of equal length
			corresponding to a joint history of (X, Y).

	Returns
	-------
	None.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	print hist
	
	counts = [word_lookup_fut[hist[0] + e_symbol[0], hist[1] + e_symbol[1]] for e_symbol in e_symbols]
	
	print counts
	
	print numpy.array(counts)/float(numpy.sum(counts))

def get_transitions(epsilon, invepsilon, e_symbols, L_max, memoryless = False):
	"""
	Get the transitions associated with the epsilon-transducer
	stored in epsilon+invepsilon, stored in the dictionary
	trans_dict.

	Parameters
	----------
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	e_symbols : list
			The emission symbols associated with (X, Y).
	L_max : int
			The maximum history length used in estimating the
			predictive distributions.
	memoryless : bool
			True if we consider an (output) memoryless transducer,
			False otherwise.

	Returns
	-------
	trans_dict : dict
			A dictionary that maps from
				(current causal state, next input/output emission symbol)
			to the next causal state state.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	states = invepsilon.keys()
	
	# Also store the allowed transitions as an edge list,
	# for easier checking. Use a dictionary of the form
	# {(from_state, emission symbol) : to_state}.
	
	trans_dict = {}

	states_to_index = {}
	index_to_states = {}

	for state_index, state in enumerate(states):
		states_to_index[state] = state_index
		index_to_states[state_index] = state

	for state in states:
		i = states_to_index[state]
		
		need_Lmax = True # Whether or not we need to use the length L_max histories in
						 # defining the transition structure.
		
		for hist in invepsilon[state].keys():
			if len(hist[0]) == L_max - 1:
				need_Lmax = False
		
		for hist in invepsilon[state].keys():
			if len(hist[0]) == L_max:
				if need_Lmax:
					for e_symbol in e_symbols:
						if memoryless:
							s = epsilon.get((hist[0][1:] + e_symbol[0], hist[1][1:] + 'n'), -1)
						else:
							s = epsilon.get((hist[0][1:] + e_symbol[0], hist[1][1:] + e_symbol[1]), -1)
				
						if s != -1:
							if memoryless:
								trans_dict[(state, (e_symbol[0], 'n'))] = s
							else:
								trans_dict[(state, e_symbol)] = s
				else:
					pass
			else:
				for e_symbol in e_symbols:
					if memoryless:
						s = epsilon.get((hist[0] + e_symbol[0], hist[1] + 'n'), -1)
					else:
						s = epsilon.get((hist[0] + e_symbol[0], hist[1] + e_symbol[1]), -1)
			
					if s != -1:
						if memoryless:
							trans_dict[(state, (e_symbol[0], 'n'))] = s
						else:
							trans_dict[(state, e_symbol)] = s
	return trans_dict



def estimate_predictive_distributions(stringX, stringY, L_max, is_multiline = False, verbose = True):
	"""
	Given a string of inputs and outputs,
	returns the counts associated with
		(xpast, ypast, xfuture)
	and with 
		(xpast, ypast, xfuture, yfuture),
	which allows us to estimate
		P(yfuture | xfuture, xpast, ypast)
	as 
		#(xpast, ypast, xfuture, yfuture)/#(xfuture, xpast, ypast)
	
	NOTE: This estimates the predictive distributions
	under the (output) memoryful assumption. Use 
	estimate_predictive_distributions_memoryless
	for the (output) memoryless implementation.

	Parameters
	----------
	stringX : str
			The string associated with the realization from the
			input process X.
	stringY : str
			The string associated with the realization from the
			output process Y.
	L_max : int
			The maximum history length to use in inferring the
			predictive distributions.
	is_multiline : bool
			True if the input files are stored with a single
			realization per line.
	verbose : bool
			True if various warning / update messages should
			be printed.
	

	Returns
	-------
	word_lookup_marg : dict
			A dictionary that maps from joint
			pasts of length <= L_max to the counts
			of the pasts.
	word_lookup_fut : dict
			A dictionary that maps from joint
			pasts + next output symbol of 
			length <= L_max to the counts of the
			number of those occurrences.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""

	if is_multiline:
		Xs = copy.copy(stringX); Ys = copy.copy(stringY)
		
		# Counter for events (X_{t-L}^{t-1}, Y_{t-L}^{t-1})

		word_lookup_marg = Counter()

		# Counter for events (X_{t-L}^{t-1}, Y_{t-L}^{t-1}, Y_{t})

		word_lookup_fut  = Counter()
		
		if verbose:
			print 'Estimating predictive distributions using multi-line.'
		
		for line_ind in range(len(Xs)):
			stringX = Xs[line_ind]; stringY = Ys[line_ind]
			
			Tx = len(stringX)
	
			Ty = len(stringY)
	
			assert Tx == Ty, 'The two time series must have the same length.'
	
			T = Tx

			for t_ind in range(T-L_max):
				cur_stringX = stringX[t_ind:(t_ind + L_max + 1)]
	
				cur_stringY = stringY[t_ind:(t_ind + L_max + 1)]
	
				word_lookup_marg[(cur_stringX, cur_stringY[:-1])] += 1
				word_lookup_fut[(cur_stringX, cur_stringY)] += 1
	
				# for remove_inds in range(0, L_max+1): DON'T NEED THIS
				for remove_inds in range(1, L_max+1):
					trunc_stringX = cur_stringX[:-remove_inds]
					trunc_stringY = cur_stringY[:-remove_inds]
		
					word_lookup_marg[(trunc_stringX, trunc_stringY[:-1])] += 1
					word_lookup_fut[(trunc_stringX, trunc_stringY)] += 1
	else:
		Tx = len(stringX)
	
		Ty = len(stringY)
	
		assert Tx == Ty, 'The two time series must have the same length.'
	
		T = Tx
	
		# Counter for events (X_{t-L}^{t-1}, Y_{t-L}^{t-1})

		word_lookup_marg = Counter()

		# Counter for events (X_{t-L}^{t-1}, Y_{t-L}^{t-1}, Y_{t})

		word_lookup_fut  = Counter()
		
		if verbose:
			print 'Estimating predictive distributions.'

		for t_ind in range(T-L_max):
			cur_stringX = stringX[t_ind:(t_ind + L_max + 1)]
	
			cur_stringY = stringY[t_ind:(t_ind + L_max + 1)]
	
			word_lookup_marg[(cur_stringX, cur_stringY[:-1])] += 1
			word_lookup_fut[(cur_stringX, cur_stringY)] += 1
	
			# for remove_inds in range(0, L_max+1): # DON'T NEED THIS
			for remove_inds in range(1, L_max+1):
				trunc_stringX = cur_stringX[:-remove_inds]
				trunc_stringY = cur_stringY[:-remove_inds]
		
				word_lookup_marg[(trunc_stringX, trunc_stringY[:-1])] += 1
				word_lookup_fut[(trunc_stringX, trunc_stringY)] += 1
	
	return word_lookup_marg, word_lookup_fut
def run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, alpha = 0.001, test_type = 'chi2', fname = None, verbose = False):
	"""
	run_transCSSR performs the CSSR algorithm, adapted for
	epsilon-transducers, to estimate the Shalizi-style
	epsilon-transducer from a given input/output data
	stream.
	
	Parameters
	----------
	word_lookup_marg : dict
			A dictionary that maps from joint
			pasts of length <= L_max to the counts
			of the pasts.
	word_lookup_fut : dict
			A dictionary that maps from joint
			pasts + next output symbol of 
			length <= L_max to the counts of the
			number of those occurrences.
	L_max : int
			The maximum history length to use in inferring the
			predictive distributions.
	axs : list
			The emission symbols associated with X.
	ays : list
			The emission symbols associated with Y.
	e_symbols : list
			The emission symbols associated with (X, Y).
	Xt_name : str
			The file name for the realization from the input
			process.
	Yt_name : string
			The file name for the realization from the output
			process.
	test_type : str
			The statistic used in the hypothesis test, one of 'chi2'
			(for the chi-squared statistic) or 'G' (for the 
			log-likelihood ratio / G statistic).
	alpha : float
			The significance level used when splitting histories
			during the homogenization step of CSSR.
	fname : str
			The filename to use when saving the .dot and .dat_results
			files.
	verbose : bool
			If true, print various progress and warning messages.

	Returns
	-------
	var1 : type
			description

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	# epsilon maps from histories to their associated causal states.

	epsilon = {}

	# invepsilon maps from causal states to the histories in the causal state.

	from collections import defaultdict

	invepsilon = defaultdict(dict)

	# The morph associated with a particular causal state.
	# That is, P(Y_{t} = y | S_{t-1} = s), for all y.
	# In the case of a binary alphabet, this reduces to
	# the probability of emitting a 1 given the causal state.

	morph_by_state = {}
	
	# The degrees of freedom for the hypothesis test. It is
	# the cardinality of the input/output alphabet minus 1.
	
	df = len(axs)*len(ays) - 1

	# Stage 1: Initialize.
	# 
	# Initialize the null string in its own state.

	num_states = 0
	
	epsilon[('', '')] = 0
	invepsilon[0][('', '')] = True
	
	morph_by_state[0] = [word_lookup_fut[('{}'.format(e_symbol[0]), '{}'.format(e_symbol[1]))] for e_symbol in e_symbols]

	num_states += 1
	
	# Stage 2: Homogenize
	#
	# Grow the joint histories by 1 into the past until
	# histories of length L_max are reached.
	
	if verbose:
		print 'Homogenizing'

	for L_cur in range(0, L_max):
		if verbose:
			print 'L_cur = {}'.format(L_cur)
			# Go through each causal state.
	
			print epsilon
	
		states = copy.deepcopy(invepsilon.keys())
	
		for state in states:
			# Grow each history associated with that state
		
			histories = copy.deepcopy(invepsilon[state])
		
			for history in histories:
				# Only need to consider those histories of length L_cur
				if len(history[0]) != L_cur:
					pass
				else:
					all_distinct = True # Whether all of histories generated 
										 # by this histories belong to different 
										 # causal states than this history. If yes,
										 # we should remove this history from the current
										 # causal state and update the distribution
										 # associated with the causal state.
				
					for ay in ays:
						for ax in axs:
							new_history = (ax + history[0], ay + history[1])
							
							does_not_occur = True
							
							for ax in axs:
								if word_lookup_marg[new_history[0] + ax, new_history[1]] != 0:
									does_not_occur = False
							
							if does_not_occur:
								pass
							else:
								if verbose:
									print "Considering the new history ({}, {})".format(new_history[0], new_history[1])
						
								morph_by_history = [word_lookup_fut[('{}{}'.format(new_history[0], e_symbol[0]), '{}{}'.format(new_history[1], e_symbol[1]))] for e_symbol in e_symbols]
					
								if verbose:
									print "This history has the morph nu(X_t | X_past) = {}".format(morph_by_history)
								
								test, pvalue = chisquared_test(morph_by_history, morph_by_state[state], axs, ays, alpha = alpha, test_type = test_type)
						
								if test == True:
									# print 'We reject that Markovian-ness, with a p-value of {}!'.format(pvalue)
							
									# Check if we should put the history in one of the existing states.
							
									is_matching_state = False
							
									if num_states == 1:
										pass
									else:
										cur_best_pvalue = -numpy.Inf
										cur_best_state = -1
									
										altstates = copy.deepcopy(invepsilon.keys())
									
										for altstate in altstates:
											if altstate == state:
												pass
											else:
												test, pvalue = chisquared_test(morph_by_history, morph_by_state[altstate], axs, ays, alpha = alpha, test_type = test_type)
										
												if pvalue > alpha:
													if pvalue > cur_best_pvalue:
														cur_best_pvalue = pvalue
														cur_best_state = altstate
									
										
										if cur_best_state == -1:
											is_matching_state = False
										else:
											is_matching_state = True
										
											epsilon[new_history] = cur_best_state
											invepsilon[cur_best_state][new_history] = True
										
											for emission_ind in range(len(e_symbols)):
												morph_by_state[cur_best_state][emission_ind] += morph_by_history[emission_ind]
											
								
									if not is_matching_state:
										# Create a new state
								
										epsilon[new_history] = num_states
										invepsilon[num_states][new_history] = True
										morph_by_state[num_states] = morph_by_history
								
										num_states += 1
								else:
									# We do not reject the null hypothesis, so we add the history
									# to its parent state.
								
									all_distinct = False
							
									epsilon[new_history] = state
									invepsilon[state][new_history] = True
								
									for emission_ind in range(len(e_symbols)):
										morph_by_state[state][emission_ind] += morph_by_history[emission_ind]
					if all_distinct:
						# All of the histories spawned by this history are in different
						# causal states, so remove this history from the current causal
						# state and update the morph associated with the causal state.
				
						invepsilon[state].pop(history)
				
				
						epsilon.pop(history)

						if len(invepsilon[state]) == 0:
							remove_state(state, epsilon, invepsilon, morph_by_state)
						else:
							old_morph = [word_lookup_fut[('{}{}'.format(history[0], e_symbol[0]), '{}{}'.format(history[1], e_symbol[1]))] for e_symbol in e_symbols]
							for output_ind in range(len(e_symbols)):
								morph_by_state[state][output_ind] -= old_morph[output_ind]
		if verbose:
			print 'For L_cur = {}, the causal states are:'.format(L_cur)
				# Go through each causal state.
			for state in invepsilon:
				print '{}: {}'.format(state, invepsilon[state])
	
	# print_transitions(epsilon, invepsilon)
	
	# Remove histories of length < L_max - 1.

	hists = copy.deepcopy(epsilon.keys())

	for hist in hists:
		if len(hist[0]) < L_max - 1:	
			state = epsilon[hist]
	
			epsilon.pop(hist)
			invepsilon[state].pop(hist)

	# Remove any states that have been made empty by
	# the removal of smaller histories.

	states = copy.deepcopy(invepsilon.keys())

	for state in states:
		if len(invepsilon[state]) == 0:
			if verbose:
				print 'Found one!'
			remove_state(state, epsilon, invepsilon, morph_by_state)

	# Save the causal states prior to any attempt at removing transients or determinizing.

	# draw_dot('transCSSR_results/mydot-nondet_transients', epsilon, invepsilon, axs, ays, L_max)
	# save_states('transCSSR_results/mydot-nondet_transients', epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	
	# Debugging the odd random channel:
	
	# tmp_state = epsilon[('111', '111')]
	#
	# for tmp in invepsilon[tmp_state]:
	# 	print_counts(tmp, word_lookup_marg, word_lookup_fut, e_symbols)
	
	# for tmp in epsilon:
	# 	print_counts(tmp, word_lookup_marg, word_lookup_fut, e_symbols)
	
	# Stage 3: Determinize.

	# Remove transient states.
	
	remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max)
	
	# Get out the current candidate CSM, prior to
	# determinizing.

	clusters, state_matrix, trans_dict, states_to_index, index_to_states = get_connected_component(epsilon, invepsilon, e_symbols, L_max)

	# draw_dot('transCSSR_results/mydot-predet', epsilon, invepsilon, axs, ays, L_max)
	# save_states('transCSSR_results/mydot-predet', epsilon, invepsilon, morph_by_state, axs, ays, L_max)

	# Check determinism 'by hand'

	# print_transitions(epsilon, invepsilon)

	recursive = False

	determinize_count = 0

	trans_for_dead_history = tuple([-1 for i in range(len(e_symbols))])

	# Remove dead histories, i.e. histories that don't transition anywhere.

	histories = copy.deepcopy(epsilon.keys())

	# for history in histories:
	# 	if len(history[0]) == L_max - 1:
	# 		trans_for_history = [epsilon.get((history[0] + e_symbol[0], history[1] + e_symbol[1]), -1) for e_symbol in e_symbols]
	# 	else:
	# 		trans_for_history = [epsilon.get((history[0][1:] + e_symbol[0], history[1][1:] + e_symbol[1]), -1) for e_symbol in e_symbols]
	#
	# 	print 'History {} (in precausal state {}) transitions like {} on {}'.format(history, epsilon[history], trans_for_history, e_symbols)
	#
	# 	if trans_for_history == trans_for_dead_history:
	# 		print 'Found a dead history!'

	while not recursive:
		if verbose:
			print 'On determinization step {}\n\n\n'.format(determinize_count)
	
		recursive = True
		states = copy.deepcopy(invepsilon.keys())
	
		for state in states:
			if verbose:
				print 'On state {}'.format(state)
			has_split = False
		
			histories = copy.deepcopy(invepsilon[state].keys())
		
			trans_to_hist = defaultdict(list) # Takes as a key the transitions occurring, and as a value those histories that
											  # that make those transitions.
										  
			# Thus, trans_to_hist stores all of the *unique* transitions
			# for each precausal state, as well as the histories
			# that make those transitions.
		
		
			for e_symbol in e_symbols:
				for hist_ind, x0 in enumerate(histories):
					if len(x0[0]) == L_max - 1:
						x0_to_state = epsilon.get((x0[0] + e_symbol[0], x0[1] + e_symbol[1]), -1)
					elif len(x0[0]) == L_max:
						x0_to_state = epsilon.get((x0[0][1:] + e_symbol[0], x0[1][1:] + e_symbol[1]), -1)
				
					if x0_to_state != -1: # We look for the first history, of any, with a non-null transition on this symbol.
						break
			
				if hist_ind == len(histories) - 1: # All of the histories don't make a transition on this symbol.
					pass
				else:
					to_split = []
			
					for pilot_ind in range(hist_ind + 1, len(histories)):
						xa = histories[pilot_ind]
			
						if len(xa[0]) == L_max - 1:
							xa_to_state = epsilon.get((xa[0] + e_symbol[0], xa[1] + e_symbol[1]), -1)
						elif len(xa[0]) == L_max:
							xa_to_state = epsilon.get((xa[0][1:] + e_symbol[0], xa[1][1:] + e_symbol[1]), -1)
			
						if xa_to_state != x0_to_state and xa_to_state != -1: # Ignore histories that have spurious non-transitions.
							to_split.append(xa)
				
							break
			
					if pilot_ind == len(histories) - 1: # All of the histories transition to the same state on this emission.
						pass
					else:  # We have found at least one history, xa, that transitions differently than x0. We now check for others.
						for new_ind in range(pilot_ind+1, len(histories)):
							xn = histories[new_ind]
					
							if len(xn[0]) == L_max - 1:
								xn_to_state = epsilon.get((xn[0] + e_symbol[0], xn[1] + e_symbol[1]), -1)
							elif len(xn[0]) == L_max:
								xn_to_state = epsilon.get((xn[0][1:] + e_symbol[0], xn[1][1:] + e_symbol[1]), -1)
					
							if xn_to_state == xa_to_state:
								to_split.append(xn) # xn transitions like xa, so put in the new state.
			
					if len(to_split) > 0: # We have found histories that need to be split.
						# print 'The pilot history is {}'.format(x0)
				
						# print 'Splitting out the history {} on emission symbol {}.\nThese transition to state {} instead of state {}'.format(to_split, e_symbol, xa_to_state, x0_to_state)
				
						recursive = False
						has_split = True
		
						# Create an empty morph.
		
						morph_by_state[num_states] = [0 for e_symbol in e_symbols]
		
						candidates = {}
				
						for candidate in to_split:
							candidates[candidate] = True
				
							# Bring along the children suffixes 
							# (grown one step into the past) of 
							# the history to be moved. Only do this
							# for histories of length L_max - 1
					
							# if len(candidate[0]) == L_max - 1:
							# 	for e_symbol in e_symbols:
							# 		child = (e_symbol[0] + candidate[0], e_symbol[1] + candidate[1])
							#
							# 		if epsilon.get(child, -1) == epsilon.get(candidate, -2):
							# 			candidates[child] = True
							# else:
							# 	pass
		
						for candidate in candidates:
							epsilon[candidate] = num_states
							invepsilon[num_states][candidate] = True

							invepsilon[state].pop(candidate)

							morph_by_history = [word_lookup_fut[('{}{}'.format(candidate[0], e_symbol[0]), '{}{}'.format(candidate[1], e_symbol[1]))] for e_symbol in e_symbols]

							for emission_ind in range(len(e_symbols)):
								morph_by_state[num_states][emission_ind] += morph_by_history[emission_ind]

								morph_by_state[state][emission_ind] -= morph_by_history[emission_ind]
						num_states += 1
				
						# print 'Made it to a breaking point.'
				
						break
			if has_split:
				# print 'Made it to a splitting point.'
				break
		
		determinize_count += 1
	
		# save_states('transCSSR_results/det_states{}'.format(determinize_count), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
		# draw_dot('transCSSR_results/det_states{}'.format(determinize_count), epsilon, invepsilon, axs, ays, L_max)
	
		# print_transitions(epsilon, invepsilon)

	if verbose:
		print 'The determinize step had to take place {} times.'.format(determinize_count)

	# Remove any states that have been made empty by
	# the removal of smaller histories.

	states = copy.deepcopy(invepsilon.keys())

	for state in states:
		if len(invepsilon[state]) == 0:
			if verbose:
				print 'Found one!'
			remove_state(state, epsilon, invepsilon, morph_by_state)

	# print_transitions(epsilon, invepsilon)

	# draw_dot('transCSSR_results/mydot-det_transients', epsilon, invepsilon, axs, ays, L_max)
	# save_states('transCSSR_results/mydot-det_transients', epsilon, invepsilon, morph_by_state, axs, ays, L_max)

	# Remove any transient states introduced by the determinization step.

	remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max)

	# draw_dot('transCSSR_results/mydot-det_recurrent', epsilon, invepsilon, axs, ays, L_max)
	# save_states('transCSSR_results/mydot-det_recurrent', epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	
	if fname == None:
		draw_dot_singlearrows('transCSSR_results/{}+{}'.format(Xt_name, Yt_name), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
		save_states('transCSSR_results/{}+{}'.format(Xt_name, Yt_name), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	else:
		draw_dot_singlearrows('transCSSR_results/{}'.format(fname), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
		save_states('transCSSR_results/{}'.format(fname), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	
	return epsilon, invepsilon, morph_by_state

def filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, memoryless = False):
	"""
	Given a realization from an input/output process
	and an epsilon-transducer for that process, 
	filter_and_predict filters the transducer-states,
	and the predictive probabilities associated with
	those states.

	Parameters
	----------
	stringX : str
			The string associated with the realization from the
			input process X.
	stringY : str
			The string associated with the realization from the
			output process Y.
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	morph_by_state : list
			The counts associated with the predictive distribution
			for a particular state.
	axs : list
			The emission symbols associated with X.
	ays : list
			The emission symbols associated with Y.
	e_symbols : list
			The emission symbols associated with (X, Y).
	L : int
			The maximum history length to use in filtering
			the input/output process.
	memoryless : bool
			If true, assume the transducer is memoryless,
			i.e. the next emission of the output process
			only depends on the past of the input process.

	Returns
	-------
	filtered_states : list
			The causal state sequence filtered
			from the input/output process.
	filtered_probs : list
			The probability that the output process
			emits a 1, given the current causal state.
	stringY_pred : str
			The predicted values of the output process,
			chosen to maximize the accurracy.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	input_lookup = {} # Create an ordering of the input alphabet $\mathcal{X}$,
					  # for indexing into the predictive distribution.
	
	for x_ind, x in enumerate(axs):
		input_lookup[x] = x_ind
	
	# Recall that, for a fixed causal state s, prob_by_state is a 
	# row-stochastic matrix. However, we store that matrix as an
	# array, where we stack *rows*. So entry W[x, y] is 
	# stored as prob_by_state[len(ays)*x + y].
	
	prob_by_state = {}
	
	for state in invepsilon:
		prob_by_state[state] = numpy.zeros(len(axs)*len(ays))
	
		for x_ind in range(len(axs)):
			counts_by_x = numpy.array(morph_by_state[state], dtype = 'float')[x_ind*len(ays):x_ind*len(ays) + len(ays)]
			prob_by_x = counts_by_x/numpy.sum(counts_by_x)
		
			prob_by_state[state][x_ind*len(ays):x_ind*len(ays) + len(ays)] = prob_by_x
		

	# Get out the transitions.

	trans_dict = get_transitions(epsilon, invepsilon, e_symbols, L, memoryless = memoryless)

	filtered_states = []
	filtered_probs  = []

	for ind in range(L-1):
		filtered_states.append(-1)
		filtered_probs.append(0.5)
		
	
	if memoryless:
		# s0 = epsilon.get((stringX[:L-1], 'n'*(L-1)), -1)
		s0 = epsilon.get((stringX[:L], 'n'*L), -1)
	else:
		# s0 = epsilon.get((stringX[:L-1], stringY[:L-1]), -1)
		s0 = epsilon.get((stringX[:L], stringY[:L]), -1)
	
	filtered_states.append(s0)
	
	# The input at the current time step (cur_x), and
	# its order in the alphabet axs.
	
	cur_x = stringX[L-1]
	cur_x_ord = input_lookup[cur_x]
	
	if s0 == -1:
		filtered_probs.append(0.5)
	else:
		filtered_probs.append(prob_by_state[s0][len(ays)*cur_x_ord + 1])

	num_ahead = 0
	
	# The second condition in this statement makes sure that
	# we don't loop *forever* in the case where we never synchronize.
	
	while s0 == -1 and len(filtered_states) < len(stringX): # In case we fail to synchronize at the first time we can...
		num_ahead += 1
		
		if memoryless:
			s0 = epsilon.get((stringX[num_ahead:L+num_ahead], 'n'*L), -1)
		else:
			s0 = epsilon.get((stringX[num_ahead:L+num_ahead], stringY[num_ahead:L+num_ahead]), -1)
	
		filtered_states.append(s0)
		
		# The input at the current time step (cur_x), and
		# its order in the alphabet axs.
	
		cur_x = stringX[L+num_ahead]
		cur_x_ord = input_lookup[cur_x]
		
		if s0 == -1:
			filtered_probs.append(0.5)
		else:
			filtered_probs.append(prob_by_state[s0][len(ays)*cur_x_ord + 1])

	for ind in range(L-1+num_ahead, len(stringX)-1):
		e_symbol = (stringX[ind], stringY[ind])
		
		if s0 == -1: # For when we need to resynchronize
			
			if memoryless:
				s1 = epsilon.get((stringX[ind-L+1:ind+1], 'n'*L), -1)
			else:
				s1 = epsilon.get((stringX[ind-L+1:ind+1], stringY[ind-L+1:ind+1]), -1)
		else:
			if memoryless:
				s1 = trans_dict.get((s0, (e_symbol[0], 'n')), -1)
			else:
				s1 = trans_dict.get((s0, e_symbol), -1)
		filtered_states.append(s1)
		
		# The input at the current time step (cur_x), and
		# its order in the alphabet axs.
			
		cur_x = stringX[ind+1]
		cur_x_ord = input_lookup[cur_x]
		
		if s1 == -1:
			filtered_probs.append(0.5)
		else:
			filtered_probs.append(prob_by_state[s1][len(ays)*cur_x_ord + 1])
	
		s0 = s1

	Y_pred = []
	
	for t_ind in range(len(filtered_states)):
		state = filtered_states[t_ind]
		if state == -1:
			y = 'N'
		else:
			# The input at the current time step (cur_x), and
			# its order in the alphabet axs.
	
			cur_x = stringX[t_ind]
			cur_x_ord = input_lookup[cur_x]
			
			y = str(ays[numpy.argmax(prob_by_state[state][cur_x_ord*len(ays):cur_x_ord*len(ays) + len(ays)])])
	
		Y_pred.append(y)
	
	stringY_pred = ''.join(Y_pred)
	
	return filtered_states, filtered_probs, stringY_pred
	
def run_tests_transCSSR(fnameX, fnameY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, L_max = None, metric = None, memoryless = False, verbose = True):
	"""
	Run various predictive tests (accuracy, precision, recall, 
	F-score, empirical total variation distance)  on the 
	input/output realization stored in fnameX/fnameY
	using the epsilon-transducer stored in epsilon+invepsilon.
	
	Parameters
	----------
	fnameX : str
			The name of the file containing the realization
			from the input process.
	fnameY : str
			The name of the file containing the realization
			from the output process.
	epsilon : dict
			A mapping from a history of length <= L_max to its
			associated causal state.
	invepsilon : dict
			A mapping from a causal state to the histories
			in that causal state.
	morph_by_state : list
			The counts associated with the predictive distribution
			for a particular state.
	axs : list
			The emission symbols associated with X.
	ays : list
			The emission symbols associated with Y.
	e_symbols : list
			The emission symbols associated with (X, Y).
	L : int
			The maximum history length used to infer
			the epsilon-transducer.
	L_max : int
			The maximum history length to consider when
			filtering. This is to ensure fair
			comparisons between epsilon-transducers of
			different lengths when choosing the history
			length using a model selection method
			like cross-validation.
	metric : bool
			The predictive metric to use, one of 
			{accuracy, precision, recall, F, tv}.
	memoryless : bool
			If true, assume the transducer is memoryless,
			i.e. the next emission of the output process
			only depends on the past of the input process.
	verbose : bool
			If true, print various warning messages.
	
	Returns
	-------
	correct_rates : numpy matrix
			The predictive scores, one per line in
			the fnameX/Y files.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	# NOTE: The filename should *already have* the suffix
	# '-tune', '-test', etc.
	
	# If a maximum L wasn't passed (i.e. we're not trying to 
	# compare CSMs on the same timeseries data), assume that
	# we want to use *all* of the timeseries in our test.

	if L_max == None:
		L_max = L
	
	datafileX = open('{}.dat'.format(fnameX))

	linesX = [line.rstrip() for line in datafileX]

	datafileX.close()
	
	datafileY = open('{}.dat'.format(fnameY))

	linesY = [line.rstrip() for line in datafileY]

	datafileY.close()
	
	assert len(linesX) == len(linesY), 'The files {} and {} must have the same length!'.format(fnameX, fnameY)
	
	correct_rates = numpy.zeros(len(linesX))

	for line_ind in range(len(linesX)):
		stringX = linesX[line_ind]; stringY = linesY[line_ind]
		state_series, predict_probs, prediction = filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L_max, memoryless = memoryless)
		
		# Originally, I had

		# ts_true = line[L_max - 1:]
		# ts_prediction = prediction[(L_max - L):]

		# here. I've changed that we always start
		# predicting after seeing L_max of the timeseries.
		# The CSM can start predicting with L_max - 1
		# of the timeseries, but this makes it easier to
		# compare across different methods.
		ts_true = stringY[L_max:]
		ts_prediction = prediction[L_max:] # This bit makes sure we predict
												 	 # on the same amount of timeseries
												 	 # regardless of L. Otherwise we 
												 	 # might artificially inflate the
												 	 # accuracy rate for large L CSMs.		
												 
		# For a given L, compute the metric rate on the tuning set.
		# Allowed metrics are 'accuracy', 'precision', 'recall', 'F'.
		
		if metric == 'tv':
			correct_rates[line_ind] = compute_metrics(ts_true, predict_probs, metric = metric)
		else:
			correct_rates[line_ind] = compute_metrics(ts_true, ts_prediction, metric = metric)

	return correct_rates

def load_transition_matrix_transducer(fname):
	"""
	Load the transition matrix for an epsilon-transducer
	stored in the .dot format.

	Parameters
	----------
	fname : string
			The filename (including the path) for a
			dot file that stores the epsilon-transducer.

	Returns
	-------
	trans_matrix : dictionary
			A lookup that maps (from_state, x, y) to
			(to_state, p). Thus, this applies the
			transition dynamic on seeing (x, y) in
			state from_state to determine to_state,
			and also returns
			P(S_{1}, Y_{1} | X_{1}, S_{0}).
	states : list
			A list storing the names of the states in
			the dot file.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	trans_matrix = {}
	
	states = {}
	
	with open(fname) as ofile:
		for line in ofile:
			if '->' in line:
				from_state = line.split(' -> ')[0]
				to_state   = line.split(' -> ')[1].split(' [')[0]
				
				states[from_state] = True
				states[to_state]   = True
				
				transitions = line.split('\"')[1].split('\l')
				
				for transition in transitions[:-1]:
					y = transition.split('|')[0]
					x = transition.split('|')[1].split(':')[0]
					
					p = float(transition.split('|')[1].split(':')[1])
					
					trans_matrix[(from_state, x, y)] = (to_state, p)
	return trans_matrix, states.keys()

def load_transition_matrix_machine(fname, inf_alg):
	"""
	Load the transition matrix for an epsilon-machine
	stored in the .dot format.

	Parameters
	----------
	fname : string
			The filename (including the path) for a
			dot file that stores the epsilon-machine.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	trans_matrix : dictionary
			A lookup that maps (from_state, x) to
			(to_state, p). Thus, this applies the
			transition dynamic on seeing x in
			state from_state to determine to_state,
			and also returns
			P(S_{1}, X_{1} | S_{0}).
	states : list
			A list storing the names of the states in
			the dot file.
	
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	trans_matrix = {}
	
	states = {}
	
	with open(fname) as ofile:
		for line in ofile:
			if '->' in line:
				from_state = line.split(' -> ')[0]
				to_state   = line.split(' -> ')[1].split(' [')[0]
				
				states[from_state] = True
				states[to_state]   = True
				
				transitions = line.split('\"')[1].split('\l')
				
				for transition in transitions[:-1]:
					if inf_alg == 'CSSR':
						x = transition.split(':')[0]
					elif inf_alg == 'transCSSR':
						x = transition.split(':')[0].split('|')[0]
					
					if r'\l' in transition:
						p = float(transition.split(':')[1].split(r'\l')[0])
					else:
						p = float(transition.split(':')[1].strip())
					
					trans_matrix[(from_state, x)] = (to_state, p)
					
	return trans_matrix, states.keys()

def compute_mixed_transition_matrix(machine_fname, transducer_fname, axs, ays, inf_alg):
	"""
	Given an epsilon-machine for the input process and an epsilon-transducer
	for the input-output process, compute_mixed_transition_matrix returns
	the transition matrix for the mixed state representation of the 
	joint process. The mixed states correspond to the direct product
	of the input causal states and the channel causal states.
	
	Note: This is *not* the minimal representation of the joint process,
	which is given by the joint epsilon-machine.

	Parameters
	----------
	machine_fname : string
			The path to the input epsilon-machine in dot format.
	transducer_fname : string
			The path to the input-output epsilon-transducer
			in dot format.
	axs : list
			The input alphabet.
	ays : list
			The output alphabet.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	P, T_states_to_index, M_states_to_index
	P : numpy array
			The transition matrix for the Markov
			chain associated with the mixed states.
	T_states_to_index : dict
			An ordered lookup for the channel
			causal states.
	M_states_to_index : dict
			An ordered lookup for the machine
			causal states.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	# Read in the transition matrices for the 
	# input process epsilon-machine and the
	# epsilon-transducer.

	T_trans, T_states = load_transition_matrix_transducer('/Users/daviddarmon/Dropbox/transfer/orc-wordmap/{}'.format(transducer_name))
	M_trans, M_states = load_transition_matrix_machine('/Users/daviddarmon/Dropbox/transfer/orc-wordmap/{}'.format(machine_name), inf_alg)

	# Determine the number of states resulting from a
	# direct product of the epsilon-machine and
	# epsilon-transducer states.

	num_mixed_states = len(T_states)*len(M_states)

	# Store the mixed state-to-mixed state transition
	# probabilities.

	# Note: We store these as P[i, j] = P(S_{1} = i | S_{0} = j),
	# e.g. p_{i<-j}, the opposite of the usual way of storing
	# transition probabilities. We do this so that we can
	# compute the *right* eigenvectors of the transition matrix
	# instead of the left eigenvectors.

	P = numpy.zeros(shape = (num_mixed_states, num_mixed_states))

	# Create an ordered lookup for the transducer and 
	# machine states.

	T_states_to_index = {}
	M_states_to_index = {}

	for s, T_state in enumerate(T_states):
		T_states_to_index[T_state] = s

	for s, M_state in enumerate(M_states):
		M_states_to_index[M_state] = s

	mixed_state_labels = []

	for ST in T_states:
		i_from = T_states_to_index[ST]
	
		T_offset_from = len(M_states)*i_from
		for SM in M_states:
			j_from = M_states_to_index[SM]
		
			M_offset_from = j_from
		
			mixed_state_labels.append((ST, SM))

	# Populate P by traversing *from* each
	# mixed state, and accumulating the probability
	# for the states transitioned *to*.

	for ST in T_states:
		i_from = T_states_to_index[ST]
	
		T_offset_from = len(M_states)*i_from
		for SM in M_states:
			j_from = M_states_to_index[SM]
		
			M_offset_from = j_from
		
			for ax in axs:
				SM_to, pM_to = M_trans.get((SM, ax), (None, 0))
			
				if SM_to != None:
					j_to = M_states_to_index[SM_to]
			
					M_offset_to = j_to
			
					for ay in ays:
						ST_to, pT_to = T_trans.get((ST, ax, ay), (None, 0))
					
						if ST_to != None:				
							i_to = T_states_to_index[ST_to]
				
							T_offset_to = len(M_states)*i_to
				
							P[T_offset_to + M_offset_to, T_offset_from + M_offset_from] += pT_to*pM_to
	
	return P, T_states_to_index, M_states_to_index

def compute_channel_states_distribution(P, M_states, T_states):
	"""
	Compute the stationary distribution for the mixed state 
	and channel causal states from the mixed state transition
	matrix P.

	Parameters
	----------
			The transition matrix for the Markov
			chain associated with the mixed states.
	M_states : list
			The causal states of the input process.
	T_states : list
			The channel causal states of the transducer.

	Returns
	-------
	stationary_dist_mixed : list
			The stationary distribution for the
			mixed states.
	stationary_dist_eT : list
			The stationary distribution for the
			channel causal states.
			
	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	# Compute the eigenvalues (L) and eigenvectors (V)
	# from the transition matrix P.

	L, V = numpy.linalg.eig(P)

	# Determine which eigenvector corresponds to the 
	# eigenvalue equal to 1. e.g. find the eigenvector
	# that is a scalar multiple of the mixed chains
	# stationary distribution.

	eig_one = numpy.where(numpy.abs(L - 1) < 0.0001)[0][0]

	stationary_dist_mixed = numpy.real(V[:, eig_one])

	# Recover the stationary distribution by forcing
	# to sum to 1.

	stationary_dist_mixed = stationary_dist_mixed/numpy.sum(stationary_dist_mixed)

	# Recover the stationary distribution over the channel
	# causal states by marginalizing the stationary
	# distribution over the mixed states.

	stationary_dist_eT = []

	for T in range(len(T_states)):
		stationary_dist_eT.append(stationary_dist_mixed[T*len(M_states):(T+1)*len(M_states)].sum())
	
	return stationary_dist_mixed, stationary_dist_eT

def compute_conditional_measures(machine_fname, transducer_fname, axs, ays, inf_alg):
	"""
	Given an epsilon-machine for the input process and an epsilon-transducer
	for the input-output process, compute_conditional_measures returns
	the conditional channel complexity $C_{X}$ and conditional entropy rate $h_{X}$.
	
	Note the dependence of these quantities on the *input* process {X_{t}}.

	Parameters
	----------
	machine_fname : string
			The path to the input epsilon-machine in dot format.
	transducer_fname : string
			The path to the input-output epsilon-transducer
			in dot format.
	axs : list
			The input alphabet.
	ays : list
			The output alphabet.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	C_X : float
			The input-dependent channel complexity.
	h_X : float
			The input-dependent entropy rate.
			
	"""
	
	P, T_states, M_states, T_trans, M_trans = compute_mixed_transition_matrix(machine_fname, transducer_fname, axs, ays, inf_alg)
	
	stationary_dist_mixed, stationary_dist_eT = compute_channel_states_distribution(P, M_states, T_states)
	
	C_X = -numpy.sum(stationary_dist_eT*numpy.log2(stationary_dist_eT))
	
	h_X = 0.
	
	for ST in T_states:
		i_from = T_states[ST]

		T_offset_from = len(M_states)*i_from
		
		for SM in M_states:
			j_from = M_states[SM]
	
			M_offset_from = j_from
			
			sum_over_x = 0.
			
			for ax in axs:
				SM_to, pM_to = M_trans.get((SM, ax), (None, 0))
		
				if SM_to != None:
					j_to = M_states[SM_to]
		
					M_offset_to = j_to
					
					sum_over_y = 0.
					
					for ay in ays:
						ST_to, pT_to = T_trans.get((ST, ax, ay), (None, 0))
				
						if ST_to != None and pT_to != 0:				
							i_to = T_states[ST_to]
			
							T_offset_to = len(M_states)*i_to
			
							sum_over_y += pT_to*numpy.log2(pT_to)
					sum_over_x += sum_over_y*pM_to
			
			h_X += sum_over_x*stationary_dist_mixed[T_offset_from + M_offset_from]
		
	h_X = -h_X
	
	return C_X, h_X

def map_words(xs, ys, trans_matrix, states):
	"""
	map_words takes in an input-output pair (x, y)
	and maps it to its associated transducer
	state under the transducer stored in 
	trans_matrix.

	Parameters
	----------
	xs : string
			The input word as a string, read
			from left (past) to right (future).
	ys : string
			The output word as a string, read
			from left (past) to right (future).
	trans_matrix : dict
			The epsilon-transducer, stored as
			a dictionary that takes
			
				(from_state, x, y)
				
			as an input, and maps to
			
				(to_state, prob).
	states : list
			A list containing the states
			associated with the epsilon-transducer.

	Returns
	-------
	s_to : string
			To state that the joint input-output
			word (xs, ys) maps to.
	admissible : boolean
			Whether the joint input-output word
			(xs, ys) is admissible under the
			transducer.
			

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	L = len(xs)
	
	admissible = False
	for s in states:
		s_to = s
		
		for t in range(L):
			x = xs[t]
			y = ys[t]
			
			s_to, p = trans_matrix.get((s_to, x, y), (None, -1))
			
			if s_to == None:
				break
			
		if s_to != None:
			admissible = True
		
		if admissible:
			break
	
	return s_to, admissible

# transducer_name = '/Users/daviddarmon/Dropbox/transfer/orc-wordmap/OddRandomChannel.dot'
transducer_name = '/Users/daviddarmon/Dropbox/transfer/orc-wordmap/FeedbackXORChannel.dot'

def generate_wordmap(transducer_fname, L = 8):
	"""
	Generate wordmap maps all pasts of length L
	their associated causal states and displays 
	this associated word map as in the figures
	from
		
		Computational Mechanics of Input-Output Processes: 
		Structured transformations and the $\epsilon$-transducer
	
	by Jim Crutchfield and Nix Barnett.

	Parameters
	----------
	transducer_fname : string
			The path to the dot file containing the
			epsilon-transducer.
	L : int
			The word length to use for the input-output
			words.

	Returns
	-------
	None

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	trans_matrix, states = load_transition_matrix_transducer('{}'.format(transducer_fname))

	# Generate all binary words of length L, and
	# test which are admissible via the epsilon-transducer.

	L_pow = 2**L

	cdict = {'red':   [(0.0, 1.0, 1.0),  # red decreases
	                   (1.0, 0.0, 0.0)],

	         'green': [(0.0, 0.0, 0.0),  # green increases
	                   (1.0, 1.0, 1.0)],

	         'blue':  [(0.0, 0.0, 0.0),  # no blue at all
	                   (1.0, 0.0, 0.0)]}

	red_green_cm = LinearSegmentedColormap('RedGreen', cdict, len(states))

	colors = cm.get_cmap(red_green_cm, len(states))

	states_to_col = {}

	for state_ind, state in enumerate(states):
		states_to_col[state] = state_ind

	for i_word in range(L_pow):
		xs = format(i_word, '0{}b'.format(L))
	
		for j_word in range(L_pow):
			ys = format(j_word, '0{}b'.format(L))
		
			s_to, admissible = map_words(xs, ys, trans_matrix, states)
		
			if admissible:
				x_numeric = 0.
				y_numeric = 0.
			
				xs_flipped = xs[::-1]
				ys_flipped = ys[::-1]
			
				for t in range(L):
					x_numeric += int(xs_flipped[t])*2.**(-(t+1))
					y_numeric += int(ys_flipped[t])*2.**(-(t+1))
				
			
				pylab.scatter(x_numeric, y_numeric, s = 3, c = colors(states_to_col[s_to]), edgecolor = 'none')
		
	pylab.xlim(xmin = 0, xmax = 1)
	pylab.ylim(ymin = 0, ymax = 1)

	pylab.axes().set_aspect('equal')

	pylab.show()

def predict_presynch_eM(stringX, machine_fname, axs, inf_alg, M_states_to_index = None, M_trans = None, stationary_dist_eM = None):
	"""
	Given an epsilon-machine and a past stringX, predict_presynch_eM
	returns the predictive distribution
		P(Xt = x | Xpast = stringX)
	potentially *before* filtering on the past synchronizes to
	one causal state.

	Parameters
	----------
	stringX : string
			The string to return the predictive distribution
			over axs for.
	machine_fname : string
			The path to the epsilon-machine in dot format.
	axs : list
			The process alphabet.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	pred_probs : numpy array
			The probability of the axs, given
			stringX.
	cur_states : list
			The current causal states the process could
			be in, given stringX.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	if stationary_dist_eM == None:
		P, M_states_to_index, M_trans = compute_eM_transition_matrix(machine_fname, axs, inf_alg = inf_alg)

		stationary_dist_mixed, stationary_dist_eM = compute_channel_states_distribution(P, {'A' : 0}, M_states_to_index)

	p_L = 0
	ps_Lp1 = [0 for x in axs]
	
	cur_states = [0 for state in M_states_to_index]

	for SM in M_states_to_index:
		from_state = SM
	
		p_state = stationary_dist_eM[M_states_to_index[SM]]
	
		for t in range(len(stringX)):
			to_state, p = M_trans.get((from_state, stringX[t]), (None, 0))
		
			if p == 0:
				p_state = 0
				break
			else:
				p_state = p_state*p
		
			from_state = to_state
	
		p_L = p_L + p_state
	
		if p_state == 0:
			pass
		else:
			if len(stringX) == 0:
				cur_states[M_states_to_index[from_state]] = 1
			else:
				cur_states[M_states_to_index[to_state]] = 1
			
			for x_ind, x in enumerate(axs):
				to_state, p = M_trans.get((from_state, x), (None, 0))
		
				p_state_Lp1 = p_state*p
		
				ps_Lp1[x_ind] = ps_Lp1[x_ind] + p_state_Lp1

	pred_probs = numpy.array(ps_Lp1)/float(p_L)

	return pred_probs, cur_states

def compute_eM_transition_matrix(machine_fname, axs, inf_alg):
	"""
	Given an epsilon-machine compute_transition_matrix returns
	the transition matrix for the causal states.

	Parameters
	----------
	machine_fname : string
			The path to the epsilon-machine in dot format.
	axs : list
			The process alphabet.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	P : numpy array
			The transition matrix for the Markov
			chain associated with the mixed states.
	M_states_to_index : dict
			An ordered lookup for the machine
			causal states.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	# Read in the transition matrices for the 
	# input process epsilon-machine and the
	# epsilon-transducer.

	M_trans, M_states = load_transition_matrix_machine('{}'.format(machine_fname), inf_alg = inf_alg)

	# Determine the number of states resulting from a
	# direct product of the epsilon-machine and
	# epsilon-transducer states.

	num_states = len(M_states)

	# Store the mixed state-to-mixed state transition
	# probabilities.

	# Note: We store these as P[i, j] = P(S_{1} = i | S_{0} = j),
	# e.g. p_{i<-j}, the opposite of the usual way of storing
	# transition probabilities. We do this so that we can
	# compute the *right* eigenvectors of the transition matrix
	# instead of the left eigenvectors.

	P = numpy.zeros(shape = (num_states, num_states))

	# Create an ordered lookup for the transducer and 
	# machine states.

	M_states_to_index = {}

	for s, M_state in enumerate(M_states):
		M_states_to_index[M_state] = s

	# Populate P by traversing *from* each
	# causal state, and accumulating the probability
	# for the states transitioned *to*.

	for SM in M_states:
		j_from = M_states_to_index[SM]
	
		M_offset_from = j_from
	
		for ax in axs:
			SM_to, pM_to = M_trans.get((SM, ax), (None, 0))
		
			if SM_to != None:
				j_to = M_states_to_index[SM_to]
		
				M_offset_to = j_to
			
				P[M_offset_to, M_offset_from] += pM_to
	
	return P, M_states_to_index, M_trans

def predict_presynch_eT(stringX, stringY, machine_fname, transducer_fname, axs, ays, inf_alg, M_states_to_index = None, T_states_to_index = None, M_trans = None, T_trans = None, stationary_dist_mixed = None, stationary_dist_eT = None):
	"""
	Given an epsilon-machine for the input process, an
	epsilon-transducer for the input-output process, 
	an input past stringX, and an output past stringY,
	predict_presynch_eT returns the predictive distribution
		P(Yt = y | Xt = stringX[-1], Xpast = stringX, Ypast = stringY)
	potentially *before* filtering on the past synchronizes to
	one causal state.

	Parameters
	----------
	stringX : string
			The input past, including the present.
	stringY : string
			The output past, not including the present.
	machine_fname : string
			The path to the epsilon-machine in dot format.
	transducer_fname : string
			The path to the epsilon-transducer in dot format.
	axs : list
			The input process alphabet.
	ays : list
			The output process alphabet.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	pred_probs : numpy array
			The probability of the ays, given
			stringX and stringY.
	cur_states : list
			The current causal states the process could
			be in, given stringX and stringY.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	
	if M_states_to_index == None or T_states_to_index == None or M_trans == None or T_trans == None or stationary_dist_mixed == None or stationary_dist_eT == None: # Only recompute these if we need to.
		P, T_states_to_index, M_states_to_index, T_trans, M_trans = compute_mixed_transition_matrix(machine_fname, transducer_fname, axs, ays, inf_alg = 'CSSR')
		
		T_states = T_states_to_index.keys()
		M_states = M_states_to_index.keys()
		
		stationary_dist_mixed, stationary_dist_eT = compute_channel_states_distribution(P, M_states, T_states)
	else:
		T_states = T_states_to_index.keys()
		M_states = M_states_to_index.keys()

	# Compute finite-L predictive probabilities:
	# 
	# P(Y_{L+1} = y_{L+1} | X_{L+1} = x_{L+1}, X_{1}^{L} = x_{1}^{L}, Y_{1}^{L} = y_{1}^{L})

	p_joint_string_L = 0.
	p_joint_string_Lp1 = [0. for y in ays]

	p_input_string_L = 0.
	p_input_string_Lp1 = 0.
	
	cur_states = [0 for state in T_states]

	for start_state_index in range(len(M_states)*len(T_states)):
		if stationary_dist_mixed[start_state_index] > 0.:
			T_start_state = T_states[int(numpy.floor(start_state_index/float(len(M_states))))]
			M_start_state = M_states[int(start_state_index - numpy.floor(start_state_index/float(len(M_states)))*len(M_states))]
	
			# Compute P(Y_{1}^{L} | X_{1}^{L}, S_{0}) and
			# P(X_{1}^{L} | S_{0})
	
			p_eT = 1.
			p_eM = 1.
	
			T_state_from = T_start_state
			M_state_from = M_start_state
	
			for t in range(len(stringX)-1):
				x = stringX[t]
				y = stringY[t]
		
				T_state_to, pT_to = T_trans.get((T_state_from, x, y), (None, 0))
		
				if pT_to == 0:
					p_eT = 0.
				else:
					p_eT = p_eT * pT_to
		
				T_state_from = T_state_to
		
				M_state_to, pM_to = M_trans.get((M_state_from, x), (None, 0))
		
				if pM_to == 0:
					p_eM = 0.
					break
				else:
					p_eM = p_eM * pM_to
		
				M_state_from = M_state_to
			
				if t == (len(stringX)-2):
					p_joint_string_L += p_eT*p_eM*stationary_dist_mixed[start_state_index]
					p_input_string_L += p_eM*stationary_dist_mixed[start_state_index]
			
			if len(stringY) == 0:
				cur_states[T_states_to_index[T_state_from]] = 1
			else:
				if p_eT != 0:
					cur_states[T_states_to_index[T_state_to]] = 1
		
			for ay_ind, ay in enumerate(ays):
				x = stringX[-1]
				y = ay
			
				T_state_to, pT_to = T_trans.get((T_state_from, x, y), (None, 0))
		
				if pT_to == 0:
					p_eT_new = 0.
				else:
					p_eT_new = p_eT * pT_to
		
				M_state_to, pM_to = M_trans.get((M_state_from, x), (None, 0))
		
				if pM_to == 0:
					p_eM_new = 0.
					break
				else:
					p_eM_new = p_eM * pM_to
		
				p_joint_string_Lp1[ay_ind] += p_eT_new*p_eM_new*stationary_dist_mixed[start_state_index]
			p_input_string_Lp1 += p_eM_new*stationary_dist_mixed[start_state_index]

	if len(stringX) == 1:
		if p_input_string_Lp1 == 0:
			# print 'This input/output pair is not allowed by the machine/transducer pair.'
		
			return [numpy.nan for y in ays], cur_states
		else:
			return (numpy.array(p_joint_string_Lp1) / p_input_string_Lp1), cur_states
	else:
		if p_input_string_Lp1 == 0 or p_input_string_L == 0 or p_joint_string_L / p_input_string_L == 0:
			# print 'This input/output pair is not allowed by the transducer.'
		
			return [numpy.nan for y in ays], cur_states
		else:
			return (numpy.array(p_joint_string_Lp1) / p_input_string_Lp1)/(p_joint_string_L / p_input_string_L), cur_states

def compute_output_transition_matrix(machine_fname, transducer_fname, axs, ays, inf_alg):
	"""
	Given an epsilon-machine for the input process and an epsilon-transducer
	for the input-output process, compute_output_transition_matrix returns
	the transition matrix for the mixed state representation of the 
	output process. The mixed states correspond to the direct product
	of the input causal states and the channel causal states.
	
	Note: This is *not* the minimal representation of the output process,
	though it can be minimized to become so.

	Parameters
	----------
	machine_fname : string
			The path to the input epsilon-machine in dot format.
	transducer_fname : string
			The path to the input-output epsilon-transducer
			in dot format.
	axs : list
			The input alphabet.
	ays : list
			The output alphabet.
	inf_alg : string
			The inference algorithm used to estimate the machine.
			One of {'CSSR', 'transCSSR'}

	Returns
	-------
	P : numpy array
			The transition matrix for the Markov
			chain associated with the mixed states.
	T_states_to_index : dict
			An ordered lookup for the channel
			causal states.
	M_states_to_index : dict
			An ordered lookup for the machine
			causal states.

	Notes
	-----
	Any notes go here.

	Examples
	--------
	>>> import module_name
	>>> # Demonstrate code here.

	"""
	# Read in the transition matrices for the 
	# input process epsilon-machine and the
	# epsilon-transducer.

	T_trans, T_states = load_transition_matrix_transducer('{}'.format(transducer_fname))
	M_trans, M_states = load_transition_matrix_machine('{}'.format(machine_fname), inf_alg = inf_alg)

	# Determine the number of states resulting from a
	# direct product of the epsilon-machine and
	# epsilon-transducer states.

	num_mixed_states = len(T_states)*len(M_states)

	# Store the mixed state-to-mixed state transition
	# probabilities.

	# Note: We store these as P[i, j] = P(S_{1} = i | S_{0} = j),
	# e.g. p_{i<-j}, the opposite of the usual way of storing
	# transition probabilities. We do this so that we can
	# compute the *right* eigenvectors of the transition matrix
	# instead of the left eigenvectors.
	
	P = {}
	
	for ay in ays:
		P[ay] = numpy.zeros(shape = (num_mixed_states, num_mixed_states))

	# Create an ordered lookup for the transducer and 
	# machine states.

	T_states_to_index = {}
	M_states_to_index = {}

	for s, T_state in enumerate(T_states):
		T_states_to_index[T_state] = s

	for s, M_state in enumerate(M_states):
		M_states_to_index[M_state] = s

	mixed_state_labels = []

	for ST in T_states:
		i_from = T_states_to_index[ST]
	
		T_offset_from = len(M_states)*i_from
		for SM in M_states:
			j_from = M_states_to_index[SM]
		
			M_offset_from = j_from
		
			mixed_state_labels.append((ST, SM))

	# Populate P by traversing *from* each
	# mixed state, and accumulating the probability
	# for the states transitioned *to*.

	for ST in T_states:
		i_from = T_states_to_index[ST]
	
		T_offset_from = len(M_states)*i_from
		for SM in M_states:
			j_from = M_states_to_index[SM]
		
			M_offset_from = j_from
		
			for ax in axs:
				SM_to, pM_to = M_trans.get((SM, ax), (None, 0))
			
				if SM_to != None:
					j_to = M_states_to_index[SM_to]
			
					M_offset_to = j_to
			
					for ay in ays:
						ST_to, pT_to = T_trans.get((ST, ax, ay), (None, 0))
					
						if ST_to != None:				
							i_to = T_states_to_index[ST_to]
				
							T_offset_to = len(M_states)*i_to
				
							P[ay][T_offset_to + M_offset_to, T_offset_from + M_offset_from] += pT_to*pM_to
	
	return P, T_states_to_index, M_states_to_index, T_trans, M_trans