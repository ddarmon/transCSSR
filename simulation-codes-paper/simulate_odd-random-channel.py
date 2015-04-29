import numpy

# The story: the odd random channel stores the parity of its
# input sequence. If the parity is even (an even number of 1s
# have been observed since the last 0), it acts as the identity
# channel, and P(Yt = x | Xt = x, Even Parity) = 1. If the parity
# is odd (an odd number of 1s have been observed since
# the last 0), it emits 0 and 1 with equal probability, e.g.
# 		P(Yt = x | Xt = x, Odd Parity) = 1/2.

T = 100000

parity = 0 	# 0 when we've observed an even number of 1s
			# since the last 0, 1 otherwise.

Xs = numpy.random.randint(2, size = T)

Xs[0] = 0

Ys = numpy.zeros(Xs.shape[0], dtype = 'int16')

for t in range(1, Xs.shape[0]):
	if Xs[t-1] == 0:
		parity = 0 # Reset the parity, since we've seen 0 1s.
	else:
		parity += 1
		
		parity = parity % 2
	
	if parity == 0:
		Ys[t] = Xs[t]
	else:
		Ys[t] = numpy.random.randint(2)

print Xs[:20]

print Ys[:20]

with open('../data/Xt_odd-random-channel.dat', 'w') as wfile:
	for sym in Xs:
		wfile.write(str(sym))

with open('../data/Yt_odd-random-channel.dat', 'w') as wfile:
	for sym in Ys:
		wfile.write(str(sym))