import numpy

Xt_name = 'coinflip'

Xt = [int(char) for char in open('data/{}.dat'.format(Xt_name)).readline().strip()]
Yt = numpy.zeros(len(Xt))

num_trailing_ones = 0

p0 = 0.35
p1 = 0.75

Us = numpy.random.rand(len(Xt))

for t in range(1, len(Xt)):
	if (num_trailing_ones % 2) == 1: # In odd 'state'
		Yt[t] = 1
		
		num_trailing_ones += 1
	elif (num_trailing_ones % 2) == 0 and Xt[t-1] == 0: # In even 'state', and saw a 0
		if Us[t] <= p0:
			Yt[t] = 1 # flip
			num_trailing_ones += 1
		else:
			Yt[t] = 0 # don't flip
			num_trailing_ones = 0
		
	elif (num_trailing_ones % 2) == 0 and Xt[t-1] == 1: # In even 'state', and saw a 1
		if Us[t] <= p1:
			Yt[t] = 1 # don't flip
			num_trailing_ones += 1
		else:
			Yt[t] = 0 # flip
			num_trailing_ones = 0

with open('data/{}_through_evenflip.dat'.format(Xt_name), 'w') as wfile:
	for emission in Yt:
		wfile.write('{}'.format(int(emission)))