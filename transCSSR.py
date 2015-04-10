from collections import Counter, defaultdict
import numpy
import scipy.stats
import pandas
import random
import os
import itertools
import string
from igraph import *
import copy

# Dependencies: numpy, scipy, pandas, igraph

from filter_data_methods import *


def chisquared_test(morph1, morph2, df, alpha = 0.001, test_type = 'G'):
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
	
	df = len(morph1) - 1
	
	if test_type == 'chi2':
		chisquared_statistic = 0

		S1 = numpy.sum(morph1)
		S2 = numpy.sum(morph2)

		K1 = numpy.sqrt(S2 / float(S1))
		K2 = 1/K1

		for ind in range(df + 1):
			numerator = K1*morph1[ind] - K2*morph2[ind]
	
			denominator = morph1[ind] + morph2[ind]
		
			if denominator == 0:
				pass
			else:
				chisquared_statistic += numerator**2/float(denominator)

		# The quantile (inverse CDF) for a standard chi-square variable

		quantile = scipy.stats.chi2.ppf
		F = scipy.stats.chi2.cdf
	
		test = (chisquared_statistic > quantile(1-alpha, df)) # If true, we reject the null hypothesis. Otherwise, we do not.
	
		pvalue = 1 - F(chisquared_statistic, df)
	elif test_type == 'G':
		chisquared_statistic = 0.
		
		n1 = numpy.sum(morph1)
		n2 = numpy.sum(morph2)

		ns = [n1, n2]

		theta0 = (numpy.array(morph1) + numpy.array(morph2))/float(n1 + n2)

		contingency_table = numpy.vstack((morph1, morph2))

		for array_ind in range(len(morph1)):
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
		
		df = len(morph1) - 1
		
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
def get_connected_component_memoryless(epsilon, invepsilon, e_symbols, L_max):
	"""
	For a finite-state Markov chain, find the largest
	component of the graph associated with the state
	transition matrix.
	
	This version works with *memoryless* transducers,
	where we assume that the output at time t depends
	only on the input at previous times.

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
						s = epsilon.get((hist[0][1:] + e_symbol[0], hist[1][1:] + 'n'), -1)
				
						if s != -1:
							j = states_to_index[s]
					
							state_matrix[i, j] = 1
						
							trans_dict[i][j] = True
				else:
					pass
			else:
				for e_symbol in e_symbols:
					s = epsilon.get((hist[0] + e_symbol[0], hist[1] + 'n'), -1)
			
					if s != -1:
						j = states_to_index[s]
				
						state_matrix[i, j] = 1
					
						trans_dict[i][j] = True
		

	g = Graph.Adjacency(state_matrix.tolist())

	clusters = g.clusters()

	return clusters, state_matrix, trans_dict, states_to_index, index_to_states
def draw_dot(fname, epsilon, invepsilon, axs, ays, L_max):
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
							
										wfile.write('{} -> {} [label = \"({}, {})\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ax, ay))
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
							
									wfile.write('{} -> {} [label = \"({}, {})\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ax, ay))
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
	
	output_lookup = {} # Create an ordering of the output alphabet $\mathcal{Y}$,
					   # for indexing into the predictive distribution.
	
	for y_ind, y in enumerate(ays):
		output_lookup[y] = y_ind
	
	prob_by_state = {}
	
	for state in invepsilon:
		counts = numpy.array(morph_by_state[state], dtype = 'float')
		prob = counts/numpy.sum(counts)
		
		prob_by_state[state] = prob
	
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
										
										W[(state, to_state)] += '{}|{}:{:.3}\\l'.format(ay, ax, prob_by_state[state][output_lookup[ay]])
							
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
									
									W[(state, to_state)] += '{}|{}:{:.3}\\l'.format(ay, ax, prob_by_state[state][output_lookup[ay]])
		
		for from_state in invepsilon.keys():
			for to_state in invepsilon.keys():
				if exists_transition.get((from_state, to_state), False):
					wfile.write('{} -> {} [label = \"{}\"];\n'.format(numeric_to_alpha(printing_lookup[from_state]), numeric_to_alpha(printing_lookup[to_state]), W[(from_state, to_state)]))
		
		wfile.write('}')
def draw_dot_memoryless(fname, epsilon, invepsilon, axs, ays, L_max):
	"""
	This function draws the .dot file associated with the 
	epsilon-transducer stored in epsilon+invepsilon.
	
	This version works with *memoryless* transducers,
	where we assume that the output at time t depends
	only on the input at previous times.

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
						for ay in ['n']:
							for ax in axs:
								to_state = epsilon.get((history[0][1:] + ax, history[1][1:] + ay), -1)

								if to_state == -1:
									pass
								else:
									if seen_transition.get((state, to_state, (ax, ay)), False):
										pass
									else:
										seen_transition[(state, to_state, (ax, ay))] = True
							
										wfile.write('{} -> {} [label = \"({}, {})\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ax, ay))
					else:
						pass
				else:
					for ay in 'n':
						for ax in axs:
							to_state = epsilon.get((history[0] + ax, history[1] + ay), -1)

							if to_state == -1:
								pass
							else:
								if seen_transition.get((state, to_state, (ax, ay)), False):
									pass
								else:
									seen_transition[(state, to_state, (ax, ay))] = True
							
									wfile.write('{} -> {} [label = \"({}, {})\"];\n'.format(numeric_to_alpha(printing_lookup[state]), numeric_to_alpha(printing_lookup[to_state]), ax, ay))
		wfile.write('\}')

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
			
			to_print = 'distribution: '
			
			prob_by_state = numpy.array(morph_by_state[state], dtype = 'float')/numpy.sum(morph_by_state[state])
			
			for emission_ind, ay in enumerate(ays):
				to_print += 'P({}) = {}\t'.format(ay, prob_by_state[emission_ind])
				
			wfile.write(to_print + '\n')
			
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
def save_states_memoryless(fname, epsilon, invepsilon, morph_by_state, axs, ays, L_max):
	"""
	Save the states associated with the epsilon-transducer to 
	a .dat_results file. The format of the .dat_results file
	follows the conventions from the CSSR code.
	
	This version works with *memoryless* transducers,
	where we assume that the output at time t depends
	only on the input at previous times.

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
			
			to_print = 'distribution: '
			
			prob_by_state = numpy.array(morph_by_state[state], dtype = 'float')/numpy.sum(morph_by_state[state])
			
			for emission_ind, ay in enumerate(ays):
				to_print += 'P({}) = {}\t'.format(ay, prob_by_state[emission_ind])
				
			wfile.write(to_print + '\n')
			
			to_print = 'transitions: '
			
			for e_symbol in e_symbols:
				sample_history = invepsilon[state].keys()[0]
				
				if len(sample_history[0]) == L_max - 1:
					to_state = epsilon.get((sample_history[0] + e_symbol[0], sample_history[1] + 'n'), 'NULL')
				else:
					to_state = epsilon.get((sample_history[0][1:] + e_symbol[0], sample_history[1][1:] + 'n'), 'NULL')
				
				to_print += 'T(({}, {})) = {}\t'.format(e_symbol[0], 'n', printing_lookup[to_state])
			
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

def print_morph_by_states(morph_by_state):
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
		morph = morph_by_state[state]
		
		tot = float(numpy.sum(morph))
		
		print state, numpy.divide(morph, tot)


def print_counts(hist):
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
	print word_lookup_marg[hist]
	
	counts = [word_lookup_fut[hist[0], hist[1] + ay] for ay in ays]
	
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
		(xpast, ypast)
	and with 
		(xpast, ypast, yfuture),
	which allows us to estimate
		P(yfuture | xpast, ypast)
	as 
		#(xpast, ypast, yfuture)/#(xpast, ypast)
	
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
				cur_stringX = stringX[t_ind:(t_ind + L_max)]
	
				cur_stringY = stringY[t_ind:(t_ind + L_max + 1)]
	
				word_lookup_marg[(cur_stringX, cur_stringY[:-1])] += 1
				word_lookup_fut[(cur_stringX, cur_stringY)] += 1
	
				# for remove_inds in range(0, L_max+1): DON'T NEED THIS
				for remove_inds in range(1, L_max+1):
					trunc_stringX = cur_stringX[:-remove_inds]
					trunc_stringY = cur_stringY[:-remove_inds]
		
					word_lookup_marg[(trunc_stringX, trunc_stringY[:-1])] += 1
					word_lookup_fut[(trunc_stringX, trunc_stringY)] += 1
			
			# DON'T NEED THIS
			
			# word_lookup_marg[('', '')] = word_lookup_fut[('', '')] # Since we double count ('', '') in the loop above, we need to for the marginal case, fix it.
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
			cur_stringX = stringX[t_ind:(t_ind + L_max)]
	
			cur_stringY = stringY[t_ind:(t_ind + L_max + 1)]
	
			word_lookup_marg[(cur_stringX, cur_stringY[:-1])] += 1
			word_lookup_fut[(cur_stringX, cur_stringY)] += 1
	
			# for remove_inds in range(0, L_max+1): # DON'T NEED THIS
			for remove_inds in range(1, L_max+1):
				trunc_stringX = cur_stringX[:-remove_inds]
				trunc_stringY = cur_stringY[:-remove_inds]
		
				word_lookup_marg[(trunc_stringX, trunc_stringY[:-1])] += 1
				word_lookup_fut[(trunc_stringX, trunc_stringY)] += 1

		# DON'T NEED THIS
		
		# word_lookup_marg[('', '')] = word_lookup_fut[('', '')] # Since we double count ('', '') in the loop above, we need to for the marginal case, fix it.
	
	return word_lookup_marg, word_lookup_fut
def estimate_predictive_distributions_memoryless(stringX, stringY, L_max, is_multiline = False, verbose = True):
	"""
	Given a string of inputs and outputs,
	returns the counts associated with
		(xpast)
	and with 
		(xpast, yfuture),
	which allows us to estimate
		P(yfuture | xpast)
	as 
		#(xpast, yfuture)/#(xpast)
	
	NOTE: This estimates the predictive distributions
	under the (output) memoryless assumption. Use 
	estimate_predictive_distributions
	for the (output) memoryful implementation.

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
			A dictionary that maps from input
			pasts of length <= L_max to the counts
			of the pasts.
	word_lookup_fut : dict
			A dictionary that maps from input
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
				cur_stringX = stringX[t_ind:(t_ind + L_max)]
	
				cur_stringY = stringY[t_ind:(t_ind + L_max + 1)]
	
				word_lookup_marg[(cur_stringX, 'n'*L_max)] += 1
				word_lookup_fut[(cur_stringX, 'n'*L_max + cur_stringY[-1])] += 1
	
				for remove_inds in range(1, L_max+1):
					trunc_stringX = cur_stringX[:-remove_inds]
					trunc_stringY = cur_stringY[:-remove_inds]
		
					word_lookup_marg[(trunc_stringX, 'n'*(L_max - remove_inds))] += 1
					word_lookup_fut[(trunc_stringX, 'n'*(L_max - remove_inds) + trunc_stringY[-1])] += 1

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
			cur_stringX = stringX[t_ind:(t_ind + L_max)]
	
			cur_stringY = stringY[t_ind:(t_ind + L_max + 1)]
	
			word_lookup_marg[(cur_stringX, 'n'*L_max)] += 1
			word_lookup_fut[(cur_stringX, 'n'*L_max + cur_stringY[-1])] += 1
	
			for remove_inds in range(1, L_max+1):
				trunc_stringX = cur_stringX[:-remove_inds]
				trunc_stringY = cur_stringY[:-remove_inds]
		
				word_lookup_marg[(trunc_stringX, 'n'*(L_max - remove_inds))] += 1
				word_lookup_fut[(trunc_stringX, 'n'*(L_max - remove_inds) + trunc_stringY[-1])] += 1
	
	return word_lookup_marg, word_lookup_fut

def run_transCSSR(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, test_type = 'G', alpha = 0.001, fname = None, verbose = False):
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
	# the cardinality of the output alphabet minus 1.
	
	df = len(ays) - 1

	# Stage 1: Initialize.
	# 
	# Initialize the null string in its own state.

	num_states = 0

	epsilon[('', '')] = 0
	invepsilon[0][('', '')] = True

	morph_by_state[0] = [word_lookup_fut[('', '{}'.format(ay))] for ay in ays]

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
						
							if word_lookup_marg[new_history] == 0:
								pass
							else:
								if verbose:
									print "Considering the new history ({}, {})".format(new_history[0], new_history[1])
						
								morph_by_history = [word_lookup_fut[('{}'.format(new_history[0]), '{}{}'.format(new_history[1], y))] for y in ays]
					
								if verbose:
									print "This history has the morph nu(X_t | X_past) = {}".format(morph_by_history)
						
								test, pvalue = chisquared_test(morph_by_history, morph_by_state[state], df, alpha = alpha, test_type = test_type)
						
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
												test, pvalue = chisquared_test(morph_by_history, morph_by_state[altstate], df, alpha = alpha, test_type = test_type)
										
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
										
											for emission_ind in range(len(ays)):
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
								
									for emission_ind in range(len(ays)):
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
							old_morph = [word_lookup_fut[('{}'.format(history[0]), '{}{}'.format(history[1], y))] for y in ays]
							for output_ind in range(len(ays)):
								morph_by_state[state][output_ind] -= old_morph[output_ind]

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
		
						morph_by_state[num_states] = [0 for ay in ays]
		
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

							morph_by_history = [word_lookup_fut[('{}'.format(candidate[0]), '{}{}'.format(candidate[1], y))] for y in ays]

							for emission_ind in range(len(ays)):
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
def run_transCSSR_memoryless(word_lookup_marg, word_lookup_fut, L_max, axs, ays, e_symbols, Xt_name, Yt_name, test_type = 'G', alpha = 0.001, fname = None, verbose = False):
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
	# the cardinality of the output alphabet minus 1.
	
	df = len(ays) - 1

	# Stage 1: Initialize.
	# 
	# Initialize the null string in its own state.

	num_states = 0

	epsilon[('', '')] = 0
	invepsilon[0][('', '')] = True

	morph_by_state[0] = [word_lookup_fut[('', '{}'.format(ay))] for ay in ays]

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
				
					for ay in ['n']:
						for ax in axs:
							new_history = (ax + history[0], ay + history[1])
						
							if word_lookup_marg[new_history] == 0:
								pass
							else:
								if verbose:
									print "Considering the new history ({}, {})".format(new_history[0], new_history[1])
						
								morph_by_history = [word_lookup_fut[('{}'.format(new_history[0]), '{}{}'.format(new_history[1], y))] for y in ays]
					
								if verbose:
									print "This history has the morph nu(X_t | X_past) = {}".format(morph_by_history)
						
								test, pvalue = chisquared_test(morph_by_history, morph_by_state[state], df, alpha = alpha, test_type = test_type)
						
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
												test, pvalue = chisquared_test(morph_by_history, morph_by_state[altstate], df, alpha = alpha, test_type = test_type)
										
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
										
											for emission_ind in range(len(ays)):
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
								
									for emission_ind in range(len(ays)):
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
							old_morph = [word_lookup_fut[('{}'.format(history[0]), '{}{}'.format(history[1], y))] for y in ays]
							for output_ind in range(len(ays)):
								morph_by_state[state][output_ind] -= old_morph[output_ind]

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

	# Stage 3: Determinize.

	# Remove transient states.

	remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max, memoryless = True)

	# Get out the current candidate CSM, prior to
	# determinizing.

	clusters, state_matrix, trans_dict, states_to_index, index_to_states = get_connected_component_memoryless(epsilon, invepsilon, e_symbols, L_max)
	
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
						x0_to_state = epsilon.get((x0[0] + e_symbol[0], x0[1] + 'n'), -1)
					elif len(x0[0]) == L_max:
						x0_to_state = epsilon.get((x0[0][1:] + e_symbol[0], x0[1][1:] + 'n'), -1)
				
					if x0_to_state != -1: # We look for the first history, of any, with a non-null transition on this symbol.
						break
			
				if hist_ind == len(histories) - 1: # All of the histories don't make a transition on this symbol.
					pass
				else:
					to_split = []
			
					for pilot_ind in range(hist_ind + 1, len(histories)):
						xa = histories[pilot_ind]
			
						if len(xa[0]) == L_max - 1:
							xa_to_state = epsilon.get((xa[0] + e_symbol[0], xa[1] + 'n'), -1)
						elif len(xa[0]) == L_max:
							xa_to_state = epsilon.get((xa[0][1:] + e_symbol[0], xa[1][1:] + 'n'), -1)
			
						if xa_to_state != x0_to_state and xa_to_state != -1: # Ignore histories that have spurious non-transitions.
							to_split.append(xa)
				
							break
			
					if pilot_ind == len(histories) - 1: # All of the histories transition to the same state on this emission.
						pass
					else:  # We have found at least one history, xa, that transitions differently than x0. We now check for others.
						for new_ind in range(pilot_ind+1, len(histories)):
							xn = histories[new_ind]
					
							if len(xn[0]) == L_max - 1:
								xn_to_state = epsilon.get((xn[0] + e_symbol[0], xn[1] + 'n'), -1)
							elif len(xn[0]) == L_max:
								xn_to_state = epsilon.get((xn[0][1:] + e_symbol[0], xn[1][1:] + 'n'), -1)
					
							if xn_to_state == xa_to_state:
								to_split.append(xn) # xn transitions like xa, so put in the new state.
			
					if len(to_split) > 0: # We have found histories that need to be split.
						# print 'The pilot history is {}'.format(x0)
				
						# print 'Splitting out the history {} on emission symbol {}.\nThese transition to state {} instead of state {}'.format(to_split, e_symbol, xa_to_state, x0_to_state)
				
						recursive = False
						has_split = True
		
						# Create an empty morph.
		
						morph_by_state[num_states] = [0 for ay in ays]
		
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

							morph_by_history = [word_lookup_fut[('{}'.format(candidate[0]), '{}{}'.format(candidate[1], y))] for y in ays]

							for emission_ind in range(len(ays)):
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

	remove_transients(epsilon, invepsilon, morph_by_state, e_symbols, L_max, memoryless = True)
	
	# draw_dot('transCSSR_results/mydot-det_recurrent', epsilon, invepsilon, axs, ays, L_max)
	# save_states('transCSSR_results/mydot-det_recurrent', epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	
	if fname == None:
		draw_dot_memoryless('transCSSR_results/{}+{}-memoryless'.format(Xt_name, Yt_name), epsilon, invepsilon, axs, ays, L_max)
		save_states_memoryless('transCSSR_results/{}+{}-memoryless'.format(Xt_name, Yt_name), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	else:
		draw_dot_memoryless('transCSSR_results/{}-memoryless'.format(fname), epsilon, invepsilon, axs, ays, L_max)
		save_states_memoryless('transCSSR_results/{}-memoryless'.format(fname), epsilon, invepsilon, morph_by_state, axs, ays, L_max)
	
	return epsilon, invepsilon, morph_by_state

def filter_and_predict_old(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, memoryless = False):
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
	
	prob_by_state = {}

	for state in morph_by_state:
		prob_by_state[state] = numpy.array(morph_by_state[state])/float(numpy.sum(morph_by_state[state]))

	# Get out the transitions.

	trans_dict = get_transitions(epsilon, invepsilon, e_symbols, L, memoryless = memoryless)

	filtered_states = []
	filtered_probs  = []

	for ind in range(L-2):
		filtered_states.append(-1)
		filtered_probs.append(0.5)
		
	
	if memoryless:
		s0 = epsilon.get((stringX[:L-1], 'n'*(L-1)), -1)
	else:
		s0 = epsilon.get((stringX[:L-1], stringY[:L-1]), -1)

	filtered_states.append(s0)
	
	if s0 == -1:
		filtered_probs.append(0.5)
	else:
		filtered_probs.append(prob_by_state[s0][1])

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
		
		if s0 == -1:
			filtered_probs.append(0.5)
		else:
			filtered_probs.append(prob_by_state[s0][1])

	for ind in range(L-1+num_ahead, len(stringX)):
		e_symbol = (stringX[ind], stringY[ind])
		
		if s0 == -1:
			
			if memoryless:
				s1 = epsilon.get((stringX[ind-L+1:ind+1], 'n'*L), -1) # For when we need to resynchronize
			else:
				s1 = epsilon.get((stringX[ind-L+1:ind+1], stringY[ind-L+1:ind+1]), -1) # For when we need to resynchronize
		else:
			if memoryless:
				s1 = trans_dict.get((s0, (e_symbol[0], 'n')), -1)
			else:
				s1 = trans_dict.get((s0, e_symbol), -1)
		filtered_states.append(s1)
		
		if s1 == -1:
			filtered_probs.append(0.5)
		else:
			filtered_probs.append(prob_by_state[s1][1])
	
		s0 = s1

	Y_pred = ['N']

	for state in filtered_states:
		if state == -1:
			y = 'N'
		else:
			y = str(ays[numpy.argmax(prob_by_state[state])])
	
		Y_pred.append(y)

	stringY_pred = ''.join(Y_pred)
	
	return filtered_states, filtered_probs, stringY_pred
	
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
	
	prob_by_state = {}

	for state in morph_by_state:
		prob_by_state[state] = numpy.array(morph_by_state[state])/float(numpy.sum(morph_by_state[state]))

	# Get out the transitions.

	trans_dict = get_transitions(epsilon, invepsilon, e_symbols, L, memoryless = memoryless)

	filtered_states = []

	for ind in range(L-1):
		filtered_states.append(-1)
	
	if memoryless:
		s0 = epsilon.get((stringX[:L], 'n'*(L)), -1)
	else:
		s0 = epsilon.get((stringX[:L], stringY[:L]), -1)

	filtered_states.append(s0)

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

	for ind in range(L+num_ahead, len(stringX)):
		e_symbol = (stringX[ind], stringY[ind])
		
		if s0 == -1:
			
			if memoryless:
				s1 = epsilon.get((stringX[ind-L+1:ind+1], 'n'*L), -1) # For when we need to resynchronize
			else:
				s1 = epsilon.get((stringX[ind-L+1:ind+1], stringY[ind-L+1:ind+1]), -1) # For when we need to resynchronize
		else:
			if memoryless:
				s1 = trans_dict.get((s0, (e_symbol[0], 'n')), -1)
			else:
				s1 = trans_dict.get((s0, e_symbol), -1)
		
		filtered_states.append(s1)
	
		s0 = s1

	Y_pred = ['N']
	filtered_probs = [0.5]

	for state in filtered_states:
		if state == -1:
			y = 'N'
			p = 0.5
		else:
			y = str(ays[numpy.argmax(prob_by_state[state])])
			p = prob_by_state[state][1]
			
		Y_pred.append(y)
		filtered_probs.append(p)

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
	
	thresholds = numpy.linspace(0, 1, 101)
	
	confusion_matrices = numpy.zeros((2, 2, len(thresholds)))

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
		predict_probs = predict_probs[L_max:]
		# For a given L, compute the metric rate on the tuning set.
		# Allowed metrics are 'accuracy', 'precision', 'recall', 'F'.
		
		if metric == 'tv':
			correct_rates[line_ind] = compute_metrics(ts_true, predict_probs, metric = metric)
		elif metric == 'AUC':
			for threshold_ind, threshold in enumerate(thresholds):
				confusion_matrices[:, :, threshold_ind] += compute_confusion_matrix(ts_true, predict_probs, threshold = threshold)
		else:
			correct_rates[line_ind] = compute_metrics(ts_true, ts_prediction, metric = metric)

	if metric == 'AUC':
		return confusion_matrices
	else:
		return correct_rates

def load_machine(fname):
	epsilon = {}
	invepsilon = defaultdict(dict)
	morph_by_state = {}
	
	with open(fname) as ofile:
		line = ofile.readline()
		
		while line != '':
			state = int(line.split(' ')[-1])
			
			morph_by_state[state] = []
		
			line = ofile.readline()
			
			while line.split(' ')[0] != 'distribution:':
				if line == ', \n': # Deal with when the null joint history is included.
					x = ''
					y = ''
				else:
					x, y = line.strip().split(', ')
				
				epsilon[(x, y)] = state
				
				invepsilon[state][(x, y)] = True
				
				line = ofile.readline()
			
			dist = line
			
			for val in dist.split(' = ')[1:]:
				morph_by_state[state].append(float(val.split('\t')[0]))
			
			for ind in range(4):
				line = ofile.readline()
	
	
	return epsilon, invepsilon, morph_by_state

def compute_props(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, memoryless = False):
	num_states = len(invepsilon)
	
	prob_by_state = {}

	for state in morph_by_state:
		prob_by_state[state] = numpy.array(morph_by_state[state])/float(numpy.sum(morph_by_state[state]))

	state_occurrences = defaultdict(int)

	total_occurrences = 0
	
	assert len(stringX) == len(stringY), "The streams for X and Y must be the same length!"
	
	if type(stringX) == list: # multiline
		for string_ind in range(len(stringX)):
			curX = stringX[string_ind]; curY = stringY[string_ind]
			
			filtered_states, filtered_probs, stringY_pred = filter_and_predict(curX, curY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, memoryless = False)
			
			for state in filtered_states:
				if state == -1:
					pass
				else:
					state_occurrences[state] += 1
					total_occurrences += 1
	else: # single line
		filtered_states, filtered_probs, stringY_pred = filter_and_predict(stringX, stringY, epsilon, invepsilon, morph_by_state, axs, ays, e_symbols, L, memoryless = False)
	
		for state in filtered_states:
			if state == -1:
				pass
			else:
				state_occurrences[state] += 1
				total_occurrences += 1
	
	state_probs = defaultdict(float)
	
	CY = 0
	
	for state in state_occurrences:
		state_probs[state] = state_occurrences[state] / float(total_occurrences)
		
		CY -= numpy.log2(state_probs[state])*state_probs[state]
	
	joint_ER = 0 # H[Y_{t} | X_{-\infty}^{t-1}, Y_{-\infty}^{t-1}]

	for state in state_probs:
		for ayind in range(len(ays)):
			Pygs = prob_by_state[state][ayind] # P(Y_{t} = y | S_{t - 1} = s)
			Ps   = state_probs[state]          # P(S_{t - 1} = s)
		
			if Pygs == 0:
				pass
			else:
				joint_ER -= Ps*Pygs*numpy.log2(Pygs)
	
	return state_probs, CY, joint_ER