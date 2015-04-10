# 
#
#	DMD, 24 June 2014

import numpy

def load_transducer(fname):
	# epsilon maps from histories
	# to causal states
	
	epsilon = {}
	
	# prob_state takes in s_{t}
	# and outputs P(X_{t+1} = 1 | S_{t} = s_{t})
	
	prob_state = {}
	
	with open(fname) as ofile:
		line = ofile.readline()
		
		while line != '':
			state_name = line.strip().split(' ')[1]
			
			line = ofile.readline()
			
			while 'P(X = 1 | S = s)' not in line:
				xt, yt = line.strip().split(', ')
				
				epsilon[(xt, yt)] = state_name
				
				line = ofile.readline()
			
			p = float(line.split(' = ')[-1])
			
			prob_state[state_name] = p
			
			line = ofile.readline()
	
	return epsilon, prob_state

trans_name = 'excite_w_refrac'

epsilon, prob_state = load_transducer('{}.trans'.format(trans_name))

Xt_name = 'coinflip'

Yt_name = '{}-{}'.format(Xt_name, trans_name)

Xt = open('../data/{}.dat'.format(Xt_name)).readline().strip()

Yt = '1'

L = len(epsilon.keys()[0][0])

for t in range(L, len(Xt)):
	xt = Xt[t-L:t]; yt = Yt[t-L:t]
	
	cur_state = epsilon[(xt, yt)]
	
	p = prob_state[cur_state]
	
	if numpy.random.rand() < p:
		Yt += '1'
	else:
		Yt += '0'

open('../data/{}.dat'.format(Yt_name), 'w').write(Yt)