import numpy as np
from random import shuffle

class BatchRank(object):
	"""Class for BatchRank"""
	def __init__(self, k, L, Model):
		super(Ranker, self).__init__()
		self.L = L
		self.k = k
		self.Model = Model
		
	def BatchRank(self):
		#number of documents
		self.d_n = len(L)
		#Map list of elemets to numbers for simplicity
		dmap = {l:i for i,l in enumerate(L)}

		#Time steps
		self.T = 1000
		#display set to hold items in k positions
		self.display = np.zeros(k)

		#Initialization for clicks and views of documents
		self.C_bl = np.zeros((2*k+1, T, self.d_n))
		self.N_bl = np.zeros((2*k+1, T, self.d_n))

		#Active batches
		self.A = set(1)
		#Highest active batch number
		self.b_max = 1
		self.I = np.zeros((2k+1, 2))
		#Positions in batch
		self.I[1] = [1,k]
		#Batch dictionary key:"get_key(b,l)", value:set of elements
		#First Batch
		bl = self.get_key(0,1)
		self.B = {bl:set(dmap[i] for i in range(len(L)))}
		#Stages
		self.l = np.zeros(2*k+1)

		for t in range(1,T+1):
			for b in A:
				self.DisplayBatch(b,t)
			for b in A:
				self.CollectClick(b,t)
			for b in A:
				self.UpdateBatch(b,t)

	def len_batch(self,b):
		return  self.I[b,1] - self.I[b,0] + 1

	def get_key(self,b,l):
		return str(b) + "," + str(l)

	def DisplayBatch(b,t):
		l = self.l[b]
		n_min = min(self.N_bl[b,l,i] for i in self.B[bl])
		len_b = len_batch(b)
		#sort them based on number of times displayed
		least_all = np.argsort(self.N_bl[b,l])
		#get only ones in current batch
		least_viewed = [least_all[i] for i in self.B[bl]]
		#random positions
		pos_rand = shuffle(range(1,len_b+1))
		#Put the items positions to be displayed
		for k in range(self.I[b,0],self.I[b,1]+1):
			self.display[k] = least_viewed[pos_rand[k-self.I[b,0]+1]]


	def CollectClicks(self,b,t):
		l = self.l[b]
		n_min = min(self.N_bl[b,l] for i in self.B[bl])
		len_b = len_batch(b)
		#click array
		cl = np.zeros(len_b)
		#get clicks
		# ? Model.click() ??

		#update number of clicks and views
		for k in range(I[b,0],I[b,1]+1):
			if self.N_bl[self.display[k]] == n_min:
				self.C_bl += cl[k]
				self.N_bl += 1

	def UpdateBatch(self,b,t):
		l = self.l[b]
		nl = 16 * pow(2,-l) * np.log(self.T)
		#Upper and Lower bound
		Up = np.array(self.b_n)
		Low = np.array(self.b_n)
		if min(self.N_bl[b,l] for i in self.B[bl]) == nl:
			for d in self.B[bl]:
				Up[d] = #UpperBound(d)
				Low[d] = #LowerBound(d)

		#sort them based on Lower Bound in descending order
		low_all = np.argsort(Low)[::-1]
		#get only ones in current batch
		bl = get_key(b,l)
		low_bound = [low_all[i] for i in self.B[bl]]
		len_b = len_batch(b)

		B_plus = set(low_bound[:len_b])
		B_minus = self.B[bl] - B_plus

		#Find a split at the position with the highest rank
		s = 0
		max_u = max(Up[i] for i in B_minus)
		for k in range(len_b):
			if Low[low_bound[k]] > max_u:
				s = k

		if s == 0 and (len(self.B[bl]) > len_b):
			#Next Elimination Stage

			#lower bound of last position in batch
			least_val = Low[low_bound[len_b-1]]

			bl_new = get_key(b,l+1)
			self.B[bl_new] = set([d for d in self.B[bl] if Up[d] > least_val])
			self.l[b] += 1
			del self.B[bl]

		elif s > 0:
			#Split

			#Create two new batches: b_max+1, b_max+2
			self.A = (self.A | set([self.b_max+1, self.b_max+2])) - set([b])

			#Parameters for batch b_max + 1
			self.I[self.b_max + 1] = [self.I[b,0], self.I[b,0] + s -1]
			bl = get_key(self.b_max+1, 0)
			self.B[bl] = B_plus
			self.l[self.b_max+1] = 0

			#Parameters for batch b_max + 2
			self.I[self.b_max + 2] = [self.I[b,0]+s, self.I[b,1]]
			bl = get_key(self.b_max+2, 0)
			self.B[bl] = B_minus
			self.l[self.b_max+2] = 0

			self.b_max += 2