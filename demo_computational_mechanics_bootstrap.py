from transCSSR import *

# Simulate a new time series:

# Length of simulated time series
N = 1000

Yt_name = 'even'

ays = ['0', '1']

transducer_fname_true = 'transCSSR_results/+{}.dot'.format(Yt_name)

stringY = simulate_eM_fast(N, transducer_fname_true, ays, 'transCSSR')

# Perform computational mechanics bootstrap:

B = 2000 # Number of bootstrap time series to generate.

boot_out = computational_mechanics_bootstrap(stringY, ays, Yt_name_inf = '{}_inf'.format(Yt_name), B = B)

# Compute true measures for original eM:

HLs, hLs, hmu, ELs, E, Cmu, etas_matrix = compute_ict_measures(transducer_fname_true, ays, inf_alg = 'transCSSR', L_max = 10, to_plot = False)

measures_true = {'Cmu' : Cmu, 'hmu' : hmu, 'E' : E}

# Check coverage of confidence intervals:

conf_level = 0.95
alpha = 1 - conf_level

for measure_name in measures_true.keys():
	ci = boot_out['Q'][measure_name]([alpha/2, 1-alpha/2])
	print('{} : {}% CI [{}, {}], true = {}'.format(measure_name, conf_level*100, ci[0], ci[1], measures_true[measure_name]))