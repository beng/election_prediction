'''
Multilayer Perceptron that predicts the 
2008 presidential elections

mlp.py
'''
import math
import random
import csv
import sys

random.seed(0)

'''
----------------------------------
	begin general help methods
----------------------------------
'''

'''
return a vector of size x and fill with 1's
'''
def vectorinit(x, val):
	return [val] * x

'''
return a matrix of size x*y and fill random values b/w v1, v2
'''
def weightinit(x, y, val1, val2):
	return [[rndweight(val1, val2)] * y for i in range(x)]
	
'''
return a random value between val1 and val2 to initialize weights
'''
def rndweight(val1, val2):
	return random.uniform(val1, val2)

'''
sigmoid activation function
'''
def sigmoid(x):
	return math.tanh(x)

'''
used to process the normalized data file and returns a 
list of the data to then test the network against
'''
def initialization(N):
	patt = []
	pat = []
	tar = []
	ii = 0
	fileinput= csv.reader(open(sys.argv[1], 'rU'),delimiter=',',quoting=csv.QUOTE_NONE)
	for row in fileinput:
		ii += 1
		if ii%2 == 0:
			inp = []
			t = []
			i = 0
			for number in row:
				if i < N:
					inp.append(float(number))
				else:
					t.append(float(number))
				i += 1
			pat.append(inp)
			tar.append(t)
			tt = []
			tt.append(inp)
			tt.append(t)
			patt.append(tt)
	return patt, pat, tar

'''
----------------------------------
	end general help methods
----------------------------------
'''

'''
-----------------------------------
	begin neural network class
-----------------------------------
'''
class NeuralNetwork:
	'''
	1) initialize 3 vectors to keep track of each of activation values 
		-input vector
		-hidden vector
		-output vector
	2) create a matix to hold weights
		-input to hidden weights
		-hidden to output weights
	3) fill weight matricies
		-use random numbers b/w -x, y for starting values
	'''
	def __init__(self, numin, numhidden, numout, train_test):
		# 1 for true
		if train_test == 1:
			self.numin = numin
			self.numhidden = numhidden
			self.numout = numout
		
			# activation initialization
			self.actin = vectorinit(self.numin, 1.0)
			self.acthidden = vectorinit(self.numhidden, 1.0)
			self.actout = vectorinit(self.numout, 1.0)
		
			a = -(6.0/math.sqrt(self.numin + self.numhidden))
			b = (6.0/math.sqrt(self.numin + self.numhidden))
		
			# weight initialization
			self.weightin = weightinit(self.numin, self.numhidden, a, b)
			self.weightout = weightinit(self.numhidden, self.numout, a, b)
		
			self.squarederror = 0.0
			
		# just testing	
		else:
			self.numin = numin
			self.numhidden = numhidden
			self.numout = numout
		
			# activation initialization
			self.actin = vectorinit(self.numin, 1.0)
			self.acthidden = vectorinit(self.numhidden, 1.0)
			self.actout = vectorinit(self.numout, 1.0)
		
			a = -(6.0/math.sqrt(self.numin + self.numhidden))
			b = (6.0/math.sqrt(self.numin + self.numhidden))
		
			# load previous weights
			self.weightin = weightinit(self.numin, self.numhidden, a, b)
			self.weightout = weightinit(self.numhidden, self.numout, a, b)
			#self.loadweights()
		
			self.squarederror = 0.0
		
	'''
	3 parts
		1) input function
			in_j = sum i = 0 to n(w_i,j * a_i)
		2) activation function
			a_j = g(in_j)
		3) output
			return a_j
	'''
	def activation(self, inputs):
		if len(inputs) != self.numin:
			raise ValueError('wrong number of inputs')
		# input function, set inputs = to actin
		for i in range(self.numin):
			self.actin[i] = inputs[i]
		
		# activation function
		for h in range(self.numhidden):
			sum = 0.0
			for i in range(self.numin):
				# sum is equivalent to in_j
				sum = sum + self.weightin[i][h] * self.actin[i]
			# sigmoid(sum)
			self.acthidden[h] = sigmoid(sum)
			
		# output same as for hidden calculations
		for o in range(self.numout):
			sum = 0.0
			for h in range(self.numhidden):
				sum = sum + self.weightout[h][o] * self.acthidden[h]
			self.actout[o] = sigmoid(sum)
	
		# return the activation output
		return self.actout
		
	'''
	-present with training set and obtain output
		-use the random weights first to get an output
	-compare output from random weights to target output
	-correct output layer weights
		w_ho = w_ho + (learning_rate*momentum*o_h)
			w_ho = weight connecting hidden to output
			o_h = output from hidden unit h
		delta_o = O_o (1-O_o)(T_o-O_o)
			O_o = output at node O of output layer
			T_o-O_o = target output for that node
	-correct input layer of weights
	-calculate the error:
	 	error = sqrt(sum from n=0 to p(T_o - O_o)^2)/p
		p = number of units in output layer
	-repeat from step 2 for each pattern in training set to complete
		an epoch
	-shuffle training set randomly 
	-repeat from step 2 until error case reached or number of epochs
	'''
	def backprop(self, expectedoutput, learningrate):
		#1.0005 otherwise decrease by 1.005
		
		deltahidden = vectorinit(self.numhidden, 0.0)
		deltaout = vectorinit(self.numout, 0.0)
		self.squarederror = 0.0		
		
		# calculate error for output
		for o in range(self.numout):
			# t_o - o_o
			error = expectedoutput[o] - self.actout[o]
			self.squarederror += error**2
			deltaout[o] = (1 - (self.actout[o])**2 ) * error
	
		# calculate error for hidden (calculates from hidden to output)
		for h in range(self.numhidden):
			# reset error so previous error values on output arent used
			error = 0.0
			for o in range(self.numout):
				error = self.weightout[h][o] * deltaout[o]
			deltahidden[h] += (1 - (self.acthidden[h])**2) * error
					
		# update output weights
		for h in range(self.numhidden):
			for o in range(self.numout):
				self.weightout[h][o] = self.weightout[h][o] + (learningrate * deltaout[o] * self.acthidden[h]) 

		# update input weights
		for i in range(self.numin):
			for h in range(self.numhidden):
				self.weightin[i][h] = self.weightin[i][h] + (learningrate * deltahidden[h] * self.actin[i])

		return self.squarederror
	
	'''
	used to train the network with a set of prediction data known as inputs. 1 iteration of i is equivalent to an epoch 
	'''	
	def trainnetwork(self, predictions, learningrate, error = 5):
		for i in range(50):
			print("**************EPOCH NUMBER ::", i, "******************")
			error = 0.0
			for p in predictions:
				inputs = p[0]
				targetoutput = p[1]
				self.activation(inputs)
				error = error + self.backprop(targetoutput, learningrate)
				print('error ',error)
	
	'''
	used to save the weights from the training phase
	'''	
	def saveweights(self):
		writer = csv.writer(open('finalweights.csv','wb'))
		out = []

		for i in range(self.numin):
			out.append(self.weightin[i])

		for o in range(self.numhidden):
			out.append(self.weightout[o])

		writer.writerow(out)

	'''
	used to load the weights from the training phase
	'''
	def loadweights(self):
		reader = csv.reader(open('finalweights.csv', 'rU'), delimiter=',',quoting=csv.QUOTE_NONE)
		
		for row in reader:
			for number in row:
				if len(row) == 4:
					# load input weights
					self.weightin = number
				else:
					# load output weights
					self.weightout = number
		
	'''
	used to test the network against a set of prediction data
	'''	
	def testnetwork(self, predictions):
		
		for p in predictions:
			out = self.activation(p[0])
			if out[0]>out[1]:
				#o won
				print('obama won by ', (out[0]*87.92+4.94), ' percent ' )
			else:
				#m won
				print('macain won by ', (out[1]*86.67+6.54), ' percent ' )
		
'''
-----------------------------------
	 end neural network class
-----------------------------------
'''


#arg1 = input file
#arg2 = to train and test or just test

datalist, inp, out = initialization(32)
if sys.argv[2] == 0:
	nn = NeuralNetwork(32,10,2, sys.argv[2])
	nn.loadweights()
#	nn.testnetwork(datalist)
else:
	nn = NeuralNetwork(32,10,2, sys.argv[2])
#	nn.trainnetwork(datalist, .5, .3)
#	nn.saveweights()
	nn.testnetwork(datalist)