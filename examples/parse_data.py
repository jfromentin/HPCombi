import matplotlib.pyplot as plt
import numpy as np


names = ['hash', 'block', 'size', 'nb_vect', 'time']
data = [[], [], [], [], [] ]

nb_par = 5
filename = 'test'
size_end = 100000
size_nb = 200
vect_end = 10000
vect_nb = 200
nb_hash = 100
kerlen_num = 1
filename = str(size_end) + str(size_nb) + str(vect_end) + str(vect_nb) + str(nb_hash) + str(kerlen_num)
with open('../build/examples/outData/'+filename+'.data', 'r') as infile:
	for line in infile:
		numbers = line.split(';')
		if len(numbers) != nb_par :
			print "Error, data is of lenght" + str(len(numbers)) + "instread of " + str(nb_par)
			print line
		else :
			for i, number in enumerate(numbers) :
				if number.isdigit() :
					data[i].append(int(number))
				else :
					data[i].append(float(number))
nb_test = len(data[0])
ex_size = data[2][int(size_nb*3.2)]
ex_vect = 1
ex_block = 2
data.append([ time/data[2][i] for i, time in enumerate(data[4]) ])
data.append([ time/data[3][i] for i, time in enumerate(data[4]) ])
nb_par += 2

# ~ block_sizes = [32, 64, 128, 256, 512, 1024]
block_sizes = [32, 64, 128, 256]
data_block_fix_vect = [ [ [data[i][j] for j in range(nb_test) if data[1][j]==block and data[3][j]==ex_vect ] for i in range(nb_par) ] for block in block_sizes ]
data_block_fix_size = [ [ [data[i][j] for j in range(nb_test) if data[1][j]==block and data[2][j]==ex_size ] for i in range(nb_par) ] for block in block_sizes ]

minHashPerSize = min( min([ data_block_fix_vect[block][5] for block in range(len(block_sizes))]) )
print 'Minimum hash time per lenght : ' + str(minHashPerSize) + ' us.'
fig0 = plt.figure(0)
fig0.canvas.set_window_title('')
ax0 = fig0.add_subplot(1, 1, 1)
for block in range(len(block_sizes)) :
	plt.plot(data_block_fix_vect[block][2], data_block_fix_vect[block][5], label='Hashing time / lenght (%s vector, %s thread/block)'%(ex_vect, block_sizes[block]))
plt.plot( data_block_fix_vect[0][2], [minHashPerSize*1.1]*len(data_block_fix_vect[0][2]) )
plt.xlabel('Function lenght')
plt.ylabel('Hashing time / lenght (us)')
plt.legend(loc='best')
ax0.set_xscale("log", nonposx='clip')
ax0.set_yscale("log", nonposx='clip')


minHashPerVect = min( min([ data_block_fix_size[block][6] for block in range(len(block_sizes))]) )
print 'Minimum hash time per vector : ' + str(minHashPerVect) + ' us.'
fig1 = plt.figure(1)
fig1.canvas.set_window_title('')
ax1 = fig1.add_subplot(1, 1, 1)
for block in range(len(block_sizes)) :
	plt.plot(data_block_fix_size[block][3], data_block_fix_size[block][6], label='Time per hash (vectors of size %s, %s thread/block)'%(ex_size, block_sizes[block]))
plt.plot( data_block_fix_size[0][3], [minHashPerVect*1.1]*len(data_block_fix_size[0][3]) )
plt.xlabel('Number of function')
plt.ylabel('Time per hash (us)')
plt.legend(loc='best')
ax1.set_xscale("log", nonposx='clip')
ax1.set_yscale("log", nonposx='clip')


# ~ np.mean(np.array(), axis=0)
# ~ np.max(np.array(), axis=0)
# ~ np.min(np.array(), axis=0)

plt.show()	
