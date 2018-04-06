import matplotlib.pyplot as plt
import numpy as np
import sys


names = ['hash', 'block', 'size', 'nb_vect', 'time']
data = [[], [], [], [], [] ]

nb_par = 5
filename = 'test'
size_end = 100000
size_nb = 200
vect_end = 10000
vect_nb = 200
nb_hash = 100

if(len(sys.argv) > 1) and sys.argv[1] in ['1', '2', '3'] :
	kerlen_num = int(sys.argv[1])
else :
	kerlen_num = 1
	
filename = str(size_end) + str(size_nb) + str(vect_end) + str(vect_nb) + str(nb_hash) + str(kerlen_num)
print 'Kernel ' + str(kerlen_num) + ', size : 16-' + str(size_end) + ' (' + str(size_nb) + ' points)' + ', number of vectors : 1-' + str(vect_end) + ' (' + str(vect_nb) + ' points).'
print 'Reading file ' + filename
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
					
block_sizes = sorted(list(set(data[1])))
nb_test = len(data[0])
ex_size = data[2][int(size_nb*len(block_sizes)*0.8)]
ex_vect = 1
ex_block = 2
data.append([ time/data[2][i] for i, time in enumerate(data[4]) ])
data.append([ time/data[3][i] for i, time in enumerate(data[4]) ])
nb_par += 2



data_block_fix_vect = [ [ [data[i][j] for j in range(nb_test) if data[1][j]==block and data[3][j]==ex_vect ] for i in range(nb_par) ] for block in block_sizes ]
data_block_fix_size = [ [ [data[i][j] for j in range(nb_test) if data[1][j]==block and data[2][j]==ex_size ] for i in range(nb_par) ] for block in block_sizes ]

minHashPerSize = min( min([ data_block_fix_vect[block][5] for block in range(len(block_sizes))]) )
output1 = 'Minimum hash time per lenght : ' + str(minHashPerSize) + ' us (' + str(ex_vect) + ' vectors).'
print output1

if kerlen_num == 1 :
	legend = ', %s thread/block'%block_sizes[block]
elif kerlen_num == 2 :
	legend = ', %s warp/block'%block_sizes[block]
else :
	legend = ''
	
	
fig0 = plt.figure(0)
plt.title(output1)
fig0.canvas.set_window_title('Kernel %d'%kerlen_num)
ax0 = fig0.add_subplot(1, 1, 1)
for block in range(len(block_sizes)) :
	plt.plot(data_block_fix_vect[block][2], data_block_fix_vect[block][5], label='Hashing time / lenght (%s vector'%ex_vect + legend +')')
plt.plot( data_block_fix_vect[0][2], [minHashPerSize*1.1]*len(data_block_fix_vect[0][2]) )
plt.xlabel('Function lenght')
plt.ylabel('Hashing time / lenght (us)')
plt.legend(loc='best')
ax0.set_xscale("log", nonposx='clip')
ax0.set_yscale("log", nonposx='clip')


minHashPerVect = min( min([ data_block_fix_size[block][6] for block in range(len(block_sizes))]) )
output2 = 'Minimum hash time per vector : ' + str(minHashPerVect) + ' us (size ' + str(ex_size) + ').'
print output2
fig1 = plt.figure(1)
plt.title(output2)
fig1.canvas.set_window_title('Kernel %d'%kerlen_num)
ax1 = fig1.add_subplot(1, 1, 1)
for block in range(len(block_sizes)) :
	plt.plot(data_block_fix_size[block][3], data_block_fix_size[block][6], label='Time per hash (vectors of size %s'%ex_size + legend +')')
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
