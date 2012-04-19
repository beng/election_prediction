This was a project I did for an AI class where I had to write my own MLP including the backpropagation algorithm from scratch. Looking back on this I see a lot of areas where functional programming techniques could be used to reduce the code size.


IMPLEMENTATION:

I initially tried to follow the backpropagation and activation algorithms found in the textbook, but found their pseudocode for the backpropagation algorithm to be more complex and be irritating to implement. I then found a multilayer perceptron tutorial here: http://www.cs.sun.ac.za/~kroon/courses/machine_learning/lecture5/mlp.pdf which gave me the necessary algorithms needed to implement the learning. Additionally, I found another textbook entitled "Applying Neural Networks, A Practical Guide" by Kevin Swingler, which was helpful in understanding the necessary formulas used in the activation section of the neural network. 

IMPLEMENTATION:

I initially tried to follow the backpropagation and activation algorithms found in the textbook, but found their pseudocode for the backpropagation algorithm to be more complex and be irritating to implement. I then found a multilayer perceptron tutorial here: http://www.cs.sun.ac.za/~kroon/courses/machine_learning/lecture5/mlp.pdf which gave me the necessary algorithms needed to implement the learning. Additionally, I found another textbook entitled "Applying Neural Networks, A Practical Guide" by Kevin Swingler, which was helpful in understanding the necessary formulas used in the activation section of the neural network. 
Neural Network Class:

	__init__(self, numin, numhidden, numout)
		This method is used to initialize the activation vectors as well as the matrices used to store the input and 
			output weights. I use vectorinit(x, val) to create a vector of size x and fill it with the specified 	
			values (in this case the values are 1.0). The weight matrices are initialized with the 
			weightinit(x,y,val1,val2) method, which returns a list of lists where the fist list is size x and the 
			sublist is of size y. I compute val1 and val2 using the following formulas:
		
			a = -(6.0/math.sqrt(self.numin + self.numhidden))
			b = (6.0/math.sqrt(self.numin + self.numhidden))
		
	activation(self, inputs):
		This method is broken down into three parts: input function, activation function, and output function
		The input function is used to set my activation inputs <- to the inputs parameter.
		The activation function has two parts, first it calculates for the inputs where the sum of activationhidden is 
			tanh(\sum{i=0}^{n} w_{i,j} * a_{i}). Second, it does the same as above, but for the output nodes. The 
			equation used is tanh(\sum{i=0}^{n} weight_{h,o} * a_{h}). It then returns the activationoutput
		
	backprop(self, expectedoutput, learningrate, momentum)
		The backpropagation algorithm works by first receiving a training set and the expected output. During the 
			first pass of the training phase the weights connecting the input to hidden and hidden to output nodes are 	
			set to random values to get some set of weights. The errors are then computed for each of the weights. The 
			input and output weight values are then updated and this process continues n times. Specific equations are 
			more detailed pseudo-code can be seen in the method documentation.
		
	trainnetwork(self, predictions, learningrate, momentum, error)
		This method is used to train my network. It works by simply calculating the activation values for my 
			predictions and then calling the backpropagation method with the target output values. It runs for n 
			epochs or until a specified threshold error is met.
	
	testnetwork(self, predictions)
		This method simply tests the network using the given input data to see who won the election

HOW TO RUN:
	To run this requires python3. First, you must run the dataset through the preprocessor, which can be done by typing "python3 preprocessor.py [input_file_name].[extension] [output_file_name].[extension]". Preferably save it as a csv file. To then test the network using a results file, type "python3 mlp.py [testing_file_name].csv 0", where 0 represents that you only want to TEST. If you would like to train the network, then type "python3 mlp.py [file_name].csv 1", where 1 represents that you want to train AND test the network. Currently, if you want to train the network, please scroll down to the bottom of the "mlp.py" file and uncomment these 2 lines by removing the # sign (LINES 308 AND 309):
	
	#	nn.trainnetwork(datalist, .5, .3)
	#	nn.saveweights()

AND THEN COMMENT THE FOLLOWING LINES OUT (LINES 304 AND 305):
	
	#	nn.loadweights()
	#	nn.testnetwork(datalist)