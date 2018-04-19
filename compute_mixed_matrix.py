from transCSSR import *

machine_fname = 'transCSSR_results/+even-exact.dot'

axs = ['0', '1']

inf_alg = 'transCSSR'

stationary_dist_eM = None

if stationary_dist_eM == None:
	P, M_states_to_index, M_trans = compute_eM_transition_matrix(machine_fname, axs, inf_alg = inf_alg)

	stationary_dist_mixed, stationary_dist_eM = compute_channel_states_distribution(P, {'A' : 0}, M_states_to_index)

