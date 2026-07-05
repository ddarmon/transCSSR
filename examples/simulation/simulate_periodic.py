Yt = '1001'*1000

with open('data/period4.dat', 'w') as wfile:
	for emission in Yt:
		wfile.write('{}'.format(int(emission)))