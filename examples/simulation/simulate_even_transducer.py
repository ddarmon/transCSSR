import numpy

Xt = [int(char) for char in open('data/even.dat').readline().strip()]
Yt = numpy.zeros(len(Xt))

num_trailing_ones = Xt[0]

# Old
# for t in range(1, len(Xt)):
# 	if (num_trailing_ones % 2) == 1 and Xt[t-1] == 0:
# 		Yt[t] = 1
# 		num_trailing_ones += 1
# 	elif (num_trailing_ones % 2) == 0 and Xt[t-1] == 0:
# 		Yt[t] = 0
# 		num_trailing_ones = 0
# 	else:
# 		Yt[t] = 1
# 		num_trailing_ones += 1

for t in range(1, len(Xt)):
	if (num_trailing_ones % 2) == 1 and Xt[t-1] == 0:
		Yt[t] = 1
		num_trailing_ones += 1
	elif (num_trailing_ones % 2) == 0 and Xt[t-1] == 0:
		Yt[t] = 0
		num_trailing_ones = 0
	else:
		Yt[t] = 1
		num_trailing_ones += 1

with open('data/even_through_even.dat', 'w') as wfile:
	for emission in Yt:
		wfile.write('{}'.format(int(emission)))