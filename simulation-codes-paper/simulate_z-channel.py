import numpy

T = 10000

Xs = numpy.random.randint(2, size = T)

Ys = numpy.zeros(Xs.shape[0], dtype = 'int16')

for t in range(0, Xs.shape[0]):
	# if Xs[t-1] == 0:
	# 	Ys[t] = 0
	# else:
	# 	Ys[t] = numpy.random.randint(2)
	
	if Xs[t] == 0:
		Ys[t] = 0
	else:
		Ys[t] = numpy.random.randint(2)

with open('../data/Xt_z-channel.dat', 'w') as wfile:
	for sym in Xs:
		wfile.write(str(sym))

with open('../data/Yt_z-channel.dat', 'w') as wfile:
	for sym in Ys:
		wfile.write(str(sym))
