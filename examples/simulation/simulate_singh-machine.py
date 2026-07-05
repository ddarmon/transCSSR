# Simulate from the two-state transducer from 
# Figure 3 of 
#
# *Learning Predictive State Representations*
#
# DMD, 190814-12-51

import numpy

N = 10000

Xt = numpy.random.randint(low = 0, high = 3, size = N)

Yt = numpy.zeros(N, dtype = 'int32')

# 0 = u
# 1 = l
# 2 = r

cur_state = 1 # Start in the left state

for t in range(1, N):
	if Xt[t-1] == 0: # Saw a 'u'
		Yt[t] = 0
	elif Xt[t-1] == 1: # Saw a 'l'
		if cur_state == 1:
			Yt[t] = 0
		elif cur_state == 2:
			cur_state = 1
			Yt[t] = 1
	elif Xt[t-1] == 2: # Saw a 'r'
		if cur_state == 1:
			cur_state = 2
			Yt[t] = 1
		elif cur_state == 2:
			Yt[t] = 0

for t in range(10):
	print '{}\t{}'.format(Xt[t], Yt[t])

Xt_name = 'tricoin'
Yt_name = '{}_through_singh-machine'.format(Xt_name)

with open('data/{}.dat'.format(Xt_name), 'w') as wfile:
	for sym in Xt:
		wfile.write(str(sym))

with open('data/{}.dat'.format(Yt_name), 'w') as wfile:
	for sym in Yt:
		wfile.write(str(sym))