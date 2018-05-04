import matplotlib.pyplot as plt
import numpy as np
import sys


def isfloat(value):
	"""
	Test if a string can be converted into a float
	"""
	try:
		float(value)
		return True
	except ValueError:
		return False

nb_gen = 6
count = 5040

if(len(sys.argv) > 2) :
	nb_gen = int(sys.argv[1])
	count = int(sys.argv[2])

	
filename = str(nb_gen) + "_" + str(count)
print 'Reading file ' + filename
sizes = []
times = []
data = dict()
with open('../build/examples/outData/' + filename + '.data', 'r') as infile:
	for line in infile:
		[size, time] = line.replace('\n', '').split(';')
		if size.isdigit() and isfloat(time) :
			data[int(size)] = float(time)
			sizes.append(int(size))
			times.append(float(time))

sorted_sizes = [key for key in sorted(data.keys())]
sorted_times = [data[key] for key in sorted_sizes]

fig0 = plt.figure(0)
plt.title('%s generators, %s transformations'%(nb_gen, count) )
fig0.canvas.set_window_title('Time')
ax0 = fig0.add_subplot(1, 1, 1)
plt.plot(sorted_sizes, sorted_times, label='')
plt.xlabel('Transformation lenght')
plt.ylabel('Time for Renner (ms)')
plt.legend(loc='best')
ax0.set_xscale("log", nonposx='clip')
# ~ ax0.set_yscale("log", nonposx='clip')

plt.show()	
