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

data_prefix = ''

Yt_name = '1mm'
# Yt_name = 'even'

ays = ['0', '1']


N = 400

Y = simulate_eM(N, 'transCSSR_results/+{}.dot'.format(Yt_name), ays, 'transCSSR')

# Uncomment to save to the specified folder:
open('simulation_outputs/{}_sim.dat'.format(Yt_name), 'w').write(Y)