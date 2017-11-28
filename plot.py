import numpy as np
import random
import matplotlib.pyplot as plt

def plot(R,title=""):
	r_len = len(R)
	x = [i for i in range(1,r_len+1)]
	plt.plot(x,R)
	plt.ylabel('Number of times in Top 5')
	plt.xlabel('Documents with increasing probablities')
	plt.title(title)
	plt.show()

f = open("top_k.txt", "r")

count = [0] * 10
for line in f:
	top = line.strip().split()
	for i in top:
		count[int(i)]+=1

plot(count,"Top 5 Elements Results (higher number -> higher probablity for click)")