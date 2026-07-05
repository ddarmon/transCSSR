import numpy
import scipy.stats
import itertools
import copy
import string
import os

from collections import Counter, defaultdict
# The legacy engine (transCSSR.py) lives at the repo root and is not an
# installed package; put the root first on sys.path so ``import transCSSR``
# resolves to it (and not, case-insensitively on macOS/Windows, to the
# installed ``transcssr`` package).
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

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