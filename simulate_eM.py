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

# Yt_name = '1mm'
Yt_name = 'even-exact'

axs = ['0', '1']
ays = ['0', '1']

e_symbols = list(itertools.product(axs, ays)) # All of the possible pairs of emission
                                              # symbols for (x, y)

N = 10000

X = simulate_eM(N, 'transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name), axs, 'transCSSR')
open('simulation_outputs/{}+{}.dat'.format(Xt_name, Yt_name), 'w').write(X)

measures = compute_conditional_measures('transCSSR_results/+null.dot', 'transCSSR_results/{}+{}.dot'.format(Xt_name, Yt_name), axs, ays, inf_alg = 'transCSSR')

print measures