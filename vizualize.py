#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]
data = np.zeros((1000,1000))

with open(filename, 'r') as f:
	lines = f.readlines()
	for line in lines:
		words = line.split()
		data[int(words[0])][int(words[1])] = long(words[2])

plt.imshow(data,interpolation='none')
plt.show()
