import numpy as np
from random import shuffle
import sys

import warnings
warnings.filterwarnings("ignore")

np.random.seed(3)
printer = 0

class BatchRanker(object):
	"""Class for BatchRank"""
	def __init__(self, L, k, Model):
		# super(Ranker, self).__init__()
		self.L = L
		self.k = k
		self.Model = Model
		# np.random.seed(3)
		print Model.attr
		
	def BatchRank(self):
		#number of documents
		self.d_n = len(L)
		#Map list of elemets to numbers for simplicity
		dmap = {l:i for i,l in enumerate(L)}
		# print dmap

		self.ara = 0
		#Time steps
		self.T = 1000
		#display set to hold items in k positions
		self.display = np.zeros(k+1, dtype=int)

		#Initialization for clicks and views of documents
		self.C_bl = np.zeros((2*k+1, self.T, self.d_n),dtype=int)
		self.N_bl = np.zeros((2*k+1, self.T, self.d_n),dtype=int)

		#Active batches
		self.A = set([1])
		#Highest active batch number
		self.b_max = 1
		self.I = np.zeros((2*k+1, 2),dtype=int)
		#Positions in batch
		self.I[1] = [1,k]
		#Batch dictionary key:"get_key(b,l)", value:set of elements
		#First Batch
		bl = self.get_key(1,0)
		self.B = {bl:set(dmap[i] for i in L)}
		#Stages
		self.l = np.zeros(2*k+1, dtype=int)

		for self.t in range(1,self.T+1):
			for b in self.A:
				self.DisplayBatch(b,self.t)
			for b in self.A:
				self.CollectClicks(b,self.t)
			for b in self.A:
				self.UpdateBatch(b,self.t)

		# print self.B, "here"
		if self.b_max == 1:
			return 1
		return -1

	def len_batch(self,b):
		return  int(self.I[b,1] - self.I[b,0] + 1)

	def get_key(self,b,l):
		return str(b) + "," + str(int(l))

	def DKL(self, p, q):
		"""Kl divergence for two Bernoulli variables"""
		if q == 0 or q == 1:
			if p == q:
				return 0
			else:
				# print "infinity"
				return float('inf')
		if p == 0:
			# print q
			return (1-p)*np.log((1-p)/(1-q))
			# print q
		if p == 1:
			return p*np.log(p/q)

		# print p,q
		return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

	def UpperBound(self,b,d,nl):
		l = self.l[b]
		c_prob = self.C_bl[b,l,d] / nl
		#get q from [c_prob,1]
		bound = np.log(self.T) + 2*np.log(np.log(self.T))
		# print bound
		q = c_prob
		# print "init ",self.DKL(c_prob,q)
		while nl * self.DKL(c_prob,q) < bound and q < 1:
			q += 0.1
		dkl = self.DKL(c_prob,q-0.1)
		# print "up:dkl ", dkl
		return nl * dkl

	def LowerBound(self,b,d,nl):
		l = self.l[b]
		c_prob = self.C_bl[b,l,d] / nl
		# print c_prob
		#get q from [c_prob,1]
		bound = np.log(self.T) + 2*np.log(np.log(self.T))
		q = 0
		while nl * self.DKL(c_prob,q) > bound and q < c_prob:
			q += 0.1
		dkl = self.DKL(c_prob,q)
		# print "low:dkl ", dkl
		return nl * dkl

	def DisplayBatch(self,b,t):
		l = self.l[b]
		bl = self.get_key(b,l)
		# print b,l, self.B[bl]
		n_min = min(self.N_bl[b,l,i] for i in self.B[bl])
		len_b = self.len_batch(b)
		#sort them based on number of times displayed
		least_all = np.argsort(self.N_bl[b,l])
		# print "least",least_all
		#get only ones in current batch
		least_viewed = [least_all[i] for i in range(self.d_n) if least_all[i] in self.B[bl]]
		# print "least view", least_viewed
		#random positions
		pos_rand = range(len_b)
		shuffle(pos_rand)
		# print "pos", pos_rand
		#Put the items positions to be displayed
		for k in range(self.I[b,0],self.I[b,1]+1):
			# print k,self.I[b,0]
			# print len(least_viewed), len(pos_rand)
			# try:
			self.display[k] = least_viewed[pos_rand[k-self.I[b,0]]]
			# except:
			# 	print self.B[bl], self.I[b]
			# 	print "Error :",k-self.I[b,0], pos_rand, least_viewed
			# 	sys.exit()
		# print "there", self.display


	def CollectClicks(self,b,t):
		l = self.l[b]
		bl = self.get_key(b,l)
		n_min = min(self.N_bl[b,l,i] for i in self.B[bl])
		len_b = self.len_batch(b)
		#click array
		cl = np.zeros(len_b)
		#get clicks
		# ? Model.click() ??
		cl = self.Model.click(self.display, self.I[b,0], self.I[b,1])
		# print "clicks ", cl
		# print "here", self.N_bl[b,l]
		#update number of clicks and views
		for k in range(self.I[b,0],self.I[b,1]+1):
			if self.N_bl[b,l,self.display[k]] == n_min:
				self.C_bl[b,l,self.display[k]] += cl[k]
				self.N_bl[b,l,self.display[k]] += 1
				# print k, self.N_bl[b,l]


	def UpdateBatch(self,b,t):
		l = self.l[b]
		nl = 16 * pow(2,-l) * np.log(self.T)
		#Upper and Lower bound
		Up = np.zeros(self.d_n)
		Low = np.zeros(self.d_n)
		# print self.d_n,len(Low)
		bl = self.get_key(b,l)
		# print self.N_bl[b,l]
		# print min(self.N_bl[b,l,i] for i in self.B[bl]), nl
		# print b,l
		if min(self.N_bl[b,l,i] for i in self.B[bl]) > nl:
			#self.C_bl[b,l]
			for d in self.B[bl]:
				Up[d] = self.UpperBound(b,d,nl)
				Low[d] = self.LowerBound(b,d,nl)

			#sort them based on Lower Bound in descending order
			low_all = np.argsort(Low)[::-1]
			# print len(low_all), len(Low)
			#get only ones in current batch
			bl = self.get_key(b,l)
			low_bound = [low_all[i] for i in range(self.d_n) if low_all[i] in self.B[bl]]
			len_b = self.len_batch(b)

			B_plus = set(low_bound[:len_b])
			B_minus = self.B[bl] - B_plus

			#Find a split at the position with the highest rank
			s = 0
			# print self.B[bl],len_b, B_plus, B_minus
			if len(B_minus) == 0:
				# print self.B[bl]
				# sys.exit()
				return
			max_u = max(Up[i] for i in B_minus)
			for k in range(len_b):
				if Low[low_bound[k]] > max_u:
					s = k

			if s == 0 and (len(self.B[bl]) > len_b):
				#Next Elimination Stage

				#lower bound of last position in batch
				least_val = Low[low_bound[len_b-1]]

				bl_new = self.get_key(b,l+1)
				# print Low, Up
				self.B[bl_new] = set([d for d in self.B[bl] if Up[d] > least_val])
				self.l[b] += 1
				# print self.B
				del self.B[bl]

			elif s > 0:
				#Split

				#Create two new batches: b_max+1, b_max+2
				self.A = (self.A | set([self.b_max+1, self.b_max+2])) - set([b])

				#Parameters for batch b_max + 1
				self.I[self.b_max + 1] = [self.I[b,0], self.I[b,0] + s -1]
				bl = self.get_key(self.b_max+1, 0)
				# self.B[bl] = B_plus
				self.B[bl] = set(low_bound[:len_b])
				self.l[self.b_max+1] = 0
				if len(self.B[bl]) == self.k:
					global printer
					if printer == 0:
						print "Done: Top K elements:", self.B[bl]
					printer = 1
					return
					# print " ".join(str(x) for x in list(self.B[bl]))
					# sys.exit()
					# print "running"

				#Parameters for batch b_max + 2
				self.I[self.b_max + 2] = [self.I[b,0]+s, self.I[b,1]]
				bl = self.get_key(self.b_max+2, 0)
				# self.B[bl] = B_minus
				self.B[bl] = set(low_bound[len_b:])
				self.l[self.b_max+2] = 0

				bl = self.get_key(b,l)
				# print self.B
				del self.B[bl]

				self.b_max += 2
				print self.B

class ClickModel(object):
	"""Generic class for ClickModel"""
	def __init__(self,docs_n):
		# super(ClickModel, self).__init__()
		self.d_n = docs_n
		# self.attr = np.random.random_sample(docs_n,)
		self.attr = np.array([0.1*i for i in range(docs_n)])
		self.docs = [i for i in range(1,docs_n+1)]


class PBM(ClickModel):
	"""Position Based Model"""
	def __init__(self, docs_n):
		# super(PBM, ClickModel).__init__()
		self.d_n = docs_n
		self.attr = np.random.random_sample(docs_n,)
		# self.attr = np.array([0.1*i for i in range(docs_n)])
		self.docs = [i for i in range(1,docs_n+1)]
		self.exam_prob = [self.rank_prob(i) for i in range(1,docs_n+1)]

	def rank_prob(self,i):
		p = 1 - i/self.d_n

	def click(self, arr, start, end):
		cl = np.zeros(len(arr))
		for i in range(start,end+1):
			try:
				prob_attr = self.attr(arr[i])
				prob_exam = self.exam_prob(i-start+1)
				prob_sel = prob_attr * prob_exam
				cl[i] = np.random.binomial(1, p=prob_sel)
			except:
				return cl
				print "Error: Document doesn't exist"

		
class CM(ClickModel):
	"""Cascading Model"""

	def click(self, arr, start, end):
		# print self.attr
		cl = np.zeros(len(arr))
		# print "here"
		for i in range(start,end+1):
			# try:
			# print "tthere"
			prob_attr = self.attr[arr[i]]
			# print prob_attr
			cl[i] = np.random.binomial(1, p=prob_attr)
			if cl[i] == 1:
				return cl
		return cl
			# except:
			# 	return cl
			# 	print "Error: Document doesn't exist"

L = [i for i in range(10)]
k = 5
Model = CM(10)
BR = BatchRanker(L,k,Model)

while 1:
	try:
		a = BR.BatchRank()
		if printer == 1:
			break
	except:
		pass