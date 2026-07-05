# Simulate from the float-reset transducer
# from Figure 5 of
#
# *Predictive State Representations*
#
# DMD, 290814-10-33

import numpy

N = 500000

Xt = numpy.random.randint(low = 0, high = 2, size = N)

Yt = numpy.zeros(N, dtype = 'int32')

state = numpy.zeros(N, dtype = 'int32')

# 0 = f(loat), move left or right with equal pro
# 1 = r(eset)

state_max = 2 # We have states 0, 1, ..., state_max, thus state_max + 1 states.

cur_state = state_max # Start in the right-most state

state[0] = cur_state

Us = numpy.random.rand(N)

for t in range(1, N):
	if cur_state == state_max:
		if Xt[t-1] == 0: # float
			if Us[t] < 0.5:
				lr = -1
			else:
				lr = 1
			
			Yt[t] = 0
			
			
			new_state = cur_state + lr
			
			if new_state > state_max:
				new_state = state_max
		elif Xt[t-1] == 1: # reset
			Yt[t] = 1
			
			new_state = state_max
	else:
		if Xt[t-1] == 0: # float
			if Us[t] < 0.5:
				lr = -1
			else:
				lr = 1
			
			Yt[t] = 0
			
			
			new_state = cur_state + lr
			
			if new_state < 0:
				new_state = 0
		elif Xt[t-1] == 1: # reset
			Yt[t] = 0
			
			new_state = state_max
			
	cur_state = new_state
	
	state[t] = cur_state

for t in range(100):
	print '{}\t{}\t{}'.format(Xt[t], Yt[t], state[t])

Xt_name = 'coinflip'
Yt_name = '{}_through_floatreset'.format(Xt_name)

with open('data/{}.dat'.format(Xt_name), 'w') as wfile:
	for sym in Xt:
		wfile.write(str(sym))

with open('data/{}.dat'.format(Yt_name), 'w') as wfile:
	for sym in Yt:
		wfile.write(str(sym))