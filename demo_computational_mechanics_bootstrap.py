import itertools

import numpy

from transCSSR_bc import *

import matplotlib.pyplot as plt

save_prefix = ''

# Length of original time series
N = 1000

# Number of bootstrap time series
B = 200

# Yt_name = 'even'
Yt_name = 'complex-csm'

axs = ['0']
ays = ['0', '1']

e_symbols = list(itertools.product(axs, ays)) # All of the possible pairs of emission
                                              # symbols for (x, y)

transducer_fname_true = 'transCSSR_results/+{}.dot'.format(Yt_name)

stringY = simulate_eM_fast(N, transducer_fname_true, ays, 'transCSSR')

alpha = 0.001

boot_out = computational_mechanics_bootstrap(stringY, ays, Yt_name_inf = '{}_inf'.format(Yt_name), B = B, alpha = 0.001)