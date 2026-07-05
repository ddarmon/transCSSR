import numpy
import ipdb

Xt_name = 'coinflip'

Xt = [int(char) for char in open('data/{}.dat'.format(Xt_name)).readline().strip()]
Yt = numpy.zeros(len(Xt))

Us = numpy.random.rand(len(Xt))

num_trailing_ones = Xt[0]

period = 3

place_in_period = 1

for t in range(1, len(Xt)):
	# ipdb.set_trace()
	
	if (num_trailing_ones % 2 == 0) and (num_trailing_ones != 0): # Kick the system every time we see an even number of 1s in the input.
		Yt[t] = 1
		
		place_in_period = 0
	else:
		if (place_in_period % period) == 0:
			Yt[t] = 0
		else:
			Yt[t] = 1
		
		place_in_period += 1
	
	if Xt[t] == 0:
		num_trailing_ones = 0
	else:
		num_trailing_ones += 1

with open('data/{}_through_periodicevenkick.dat'.format(Xt_name), 'w') as wfile:
	for emission in Yt:
		wfile.write('{}'.format(int(emission)))