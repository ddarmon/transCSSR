import numpy
import ipdb

Xt_name = 'coinflip'

Xt = [int(char) for char in open('data/{}.dat'.format(Xt_name)).readline().strip()]
Yt = numpy.zeros(len(Xt))

Us = numpy.random.rand(len(Xt))

period = 3

place_in_period = 1

for t in range(1, len(Xt)):
	# print 'Place in period: {}'.format(place_in_period)
	
	if Xt[t-1] == 1: # Kick the system every time the previous symbol in the input was 1.
		Yt[t] = 1
		
		# print 'Kicked!'

		place_in_period = 0
	else:
		if (place_in_period % period) == 0:
			Yt[t] = 0
		else:
			Yt[t] = 1

		place_in_period += 1

with open('data/{}_through_periodickick.dat'.format(Xt_name), 'w') as wfile:
	for emission in Yt:
		wfile.write('{}'.format(int(emission)))