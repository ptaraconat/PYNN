import numpy as np 
import matplotlib.pyplot as plt 

def dsigmoid(x):
	return sigmoid(x)*(1.0-sigmoid(x))
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def dlinear(x):
	return 1
def linear(x):
	return x
	
#def RMSE(Y,Yhat):
#	res = np.power(Yhat - Y,2)
#	res = np.sum(res,axis = 0,keepdims = True) 
#	N = np.size(Yhat,0)
#	res = res/N
#	res = np.sqrt(res)
#	return res
#def DRMSE(Y,Yhat):
#	top = Y - Yhat
#	bot = np.sqrt(2*top)
#	return top/bot
	
def mean_squared_error(y,yhat):
	'''
	Inputs 
	y ::: numpy.array (Output_Dim,Batch_Size) ::: Predicted Values 
	yhat ::: numpy.array (Output_Dim,Batch_Size) ::: Targeted Values 
	Outputs 
	res::: float ::: Mean Squared Error between Y and Yhat 
	'''
	res = np.power(yhat - y,2)
	res = np.sum(res,axis = 0,keepdims = True) 
	N = np.size(yhat,0)
	res = res/N
	return res

def d_mean_squared_error(y,yhat):
	'''
	Inputs 
	y ::: numpy.array (Output_Dim,Batch_Size) ::: Predicted Values 
	yhat ::: numpy.array (Output_Dim,Batch_Size) ::: Targeted Values 
	Outputs 
	res ::: float ::: Derivative of MSE with respect to Y 
	'''
	N = np.size(yhat,0)
	return (2/N)*(y-yhat)

class Optimizer():
	
	def __init__(self,type = None):
		self.type = None

class Adam(Optimizer):
	def __init__(self,learning_rate,beta1, beta2, epsilon):
		self.ite = 1 
		self.algorithm = 'Adam'
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.learning_rate = learning_rate
	
	def update_layer(self,layer):
		'''
		Inputs
		layer ::: NN_numerics.Layer :::
		Outputs 
		None
		'''
		ite = self.ite
		beta1 = self.beta1
		beta2 = self.beta2
		epsilon = self.epsilon
		learning_rate = self.learning_rate
		# Update Sd
		layer.cache['Sdw'] = beta2*layer.cache['Sdw'] + (1-beta2)*np.square(layer.dweights_)
		layer.cache['Sdb'] = beta2*layer.cache['Sdb'] + (1-beta2)*np.square(layer.dbias_)
		# Update Vd
		layer.cache['Vdw'] = beta1*layer.cache['Vdw'] + (1-beta1)*layer.dweights_ 
		layer.cache['Vdb'] = beta1*layer.cache['Vdb'] + (1-beta1)*layer.dbias_ 
		# Correct Vd and Sd 
		Vdwc_tmp = layer.cache['Vdw']/(1-(beta1**ite))
		Vdbc_tmp = layer.cache['Vdb']/(1-(beta1**ite))
		Sdwc_tmp = layer.cache['Sdw']/(1-(beta2**ite))
		Sdbc_tmp = layer.cache['Sdb']/(1-(beta2**ite))
		#Update Model weights, using Vd and Sd: to be completed 
		layer.weights =  layer.weights - learning_rate * np.divide(Vdwc_tmp,(np.sqrt(Sdwc_tmp)+epsilon))
		layer.bias = layer.bias - learning_rate*np.divide(Vdbc_tmp,(np.sqrt(Sdbc_tmp)+epsilon))

class SGD(Optimizer):
	def __init__(self,learning_rate):
		self.ite = 1 
		self.algorithm = 'SGD'
		self.learning_rate = learning_rate

	def update_layer(self,layer):
		'''
		Inputs
		layer ::: NN_numerics.Layer :::
		Outputs 
		None
		'''
		learning_rate = self.learning_rate
		layer.weights = layer.weights - learning_rate*layer.dweights_
		layer.bias = layer.bias - learning_rate*layer.dbias_
		

class Model():
	def __init__(self,layers = [],loss = 'MSE'):
		'''
		Inputs 
		layers ::: list of NN_numerics.Layers ::: default = [] ::: Sequence of layers defining the model 
		loss ::: str ::: default = 'MSE' (choose among : 'MSE', '' ...) ::: Loss function used during the model training  
		Outputs 
		None
		'''
		self.layers = layers 
		if loss == 'MSE' : 
			self.loss = mean_squared_error
			self.dloss = d_mean_squared_error
	
	def predict(self,inputs): 
		'''
		Inputs 
		inputs ::: np.array (Input_Dim,Batch_Size) ::: Input data, for which we want to make a prediction
		Outputs 
		input_tmp ::: np.array (Output_Dim, Batch_Size) ::: Model Predictions
		'''
		input_tmp = inputs 
		## Forward layers loop 
		for layer in self.layers :
			## Propagate input_tmp through the layer  
			layer.forward(input_tmp)
			## Update input_tmp with the layer activations, stored in layer.cache 
			input_tmp = layer.cache['A']
		return input_tmp
	
	def backprop(self,y,yhat):
		'''
		Inputs  
		y ::: numpy.array (Output_Dim,Batch_Size) ::: Predicted Values 
		yhat ::: numpy.array (Output_Dim,Batch_Size) ::: Targeted Values
		Outputs 
		None
		'''
		batch_size = np.size(y,1) 
		rhs_feed = self.dloss(y,yhat) # Init right hand side feed 
		## Backward layers loop
		l = len(self.layers) - 1 
		while l >= 0: 
			## Back-propagate and update rhs_feed 
			rhs_feed = self.layers[l].backward(rhs_feed,batch_size)
			l = l - 1 
	
	def update(self,optimizer):
		'''
		Update model weights/biases 
		Inputs 
		optimizer ::: NN_numerics.Optimizer ::: Optimization algorithm 
		Outputs 
		None
		'''
		## Forward layers loop 
		l = 0
		while l < len(self.layers):
			## Update the layer 
			optimizer.update_layer(self.layers[l])
			l = l + 1
		
	def calc_loss(self,y,yhat): 
		'''
		Inputs 
		y ::: numpy.array (Output_Dim,Batch_Size) ::: Predicted Values 
		yhat ::: numpy.array (Output_Dim,Batch_Size) ::: Targeted Values 
		Outputs 
		loss ::: float ::: Mean Squared Error between Y and Yhat 
		'''
		batch_size = np.size(y,1) # Assess batch_size
		err = self.loss(y,yhat) # Assess loss 
		loss = np.sum(err)/batch_size # Normalize loss according to the number of examples 
		return loss
	
	def step(self,x_mb,y_mb): 
		'''
		Performs forward propagation and back ward propagation 
		arguments 
		x_mb ::: array (n_features,batch_size) ::: input data 
		y_mb ::: array (n_labels, batch_size) ::: output data 
		returns 
		loss ::: float ::: loss value 
		'''
		## Forward prop
		yhat = self.predict(x_mb)
		## Calc Loss
		loss = self.calc_loss(yhat,y_mb)
		## Back prop
		self.backprop(yhat,y_mb)
		return loss
		
	def fit(self,x,yhat,optimizer = Adam(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8),epochs=1000,batch_size = None):
		'''
		Inputs 
		x ::: numpy.array (Input_Dim,N_examples) ::: Input Values 
		yhat ::: numpy.array (Output_Dim,N_examples) ::: Targeted Values
		optimizer ::: NN_numerics.Adam(Optimizer) ::: default 
		= Adam(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8) ::: 
		Epochs ::: int ::: default = 1000 ::: Number of epochs 
		batch_size ::: int ::: default = None ::: Mini-batch size 
		Outputs 
		loss ::: float ::: Mean Squared Error between Y and Yhat 
		'''
		n_examples = np.size(x,-1)
		if batch_size == None :
			batch_size = n_examples
		if batch_size > n_examples:
			batch_size = n_examples
		## Training Loop 
		i = 0 
		err = []
		while (i < epochs) :
			print( 'Epoch ',str(i),' /',str(epochs))
			## Shuffle training examples 
			shuffled_indices = np.random.permutation(n_examples)
			x = x[:,shuffled_indices]
			yhat = yhat[:,shuffled_indices]
			## Mini-batches Loop
			j = 0 
			while j < n_examples:
				## Get mini-batch from shuffled training data 
				if j+batch_size < n_examples:
					x_mb = x[:,j:j+batch_size]
					yhat_mb = yhat[:,j:j+batch_size] 
				else : 
					x_mb = x[:,j:]
					yhat_mb = yhat[:,j:]
				## Forward and Backward prop 
				loss = self.step(x_mb,yhat_mb)
				## Update parameters 
				self.update(optimizer)
				optimizer.ite = optimizer.ite + 1 
				## Update error curve with loss 
				err = err + [loss]
				#print(j,'/',np.size(x,-1),'::: Mini batch Loss = ',str(loss))
				j = j + batch_size
			print( 'Last loss at Epoch ',str(i),' = ',str(loss))
			del loss
			i = i + 1 
		return err
		
	def summary(self):
		'''
		Display the model architecture 
		Inputs 
		None 
		Outputs 
		None
		'''
		print('')
		print('##### Model Summary #####')
		for i in range(len(self.layers)):
			print('###########################')
			print('Layer Number '+str(i+1))
			print('Weights ::: ',np.shape(self.layers[i].weights))
			print('Biases  ::: ',np.shape(self.layers[i].bias))
			
	def disp_learnable(self):
		'''
		Display the model parameters 
		Inputs 
		None 
		Outputs 
		None
		'''
		print('')
		print('##### Model Parameters #####')
		for i in range(len(self.layers)):
			print('###########################')
			print('Layer Number '+str(i+1))
			print('Weights ::: ')
			print(self.layers[i].weights)
			print('Biases  ::: ')
			print(self.layers[i].bias)
		

class Layer():
	def __init__(self):
		self.Name = None

class Dense(Layer):
	def __init__(self,units,activation ='linear',input_units = None): 
		'''
		Inputs 
		units ::: int ::: Number of units (viz. neurons) in the layer 
		activation ::: str ::: default = 'linear' choices('linear', 'sigmoid', '', ...) ::: Activation function 
		input_units ::: int ::: default = None ::: Number of inputs of the layer
		Outputs 
		None 
		'''
		self.units = units
		self.cache = dict() # Init Cache dictionary 
		self.cache['Sdb'] = np.zeros((self.units,1)) # Init 
		self.cache['Vdb'] = np.zeros((self.units,1)) # Init 
		self.bias = np.zeros((units,1)) + np.random.normal(loc=0.0, scale=1.0,size = (units,1)) # Init Biases 
		self.dbias_ = np.zeros((units,1)) # Init DLoss/DB
		## Set the activation 
		if activation == 'linear':
			self.activation = linear
			self.dactivation = dlinear
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.dactivation = dsigmoid
		## If the number of inputs is provided, then init weights
		if input_units != None : 
			self.input_units = input_units
			self.weights = np.zeros((self.units,self.input_units)) + np.random.normal(loc=0.0, scale=1.0,size = (self.units,self.input_units))
			self.dweights_ = np.zeros((self.units,self.input_units)) # Init DLoss/DW
			self.cache['Sdw'] = np.zeros((self.units,self.input_units))
			self.cache['Vdw'] = np.zeros((self.units,self.input_units))
			
	def forward(self,input): 
		'''
		Inputs 
		input ::: numpy.array (Prev_Layer_Dim,Batch_Size) ::: Left hand side
		feed of the layer : Activations from the previous layer, or model
		input, if this is the first layer of the model. 
		Outputs 
		None 
		'''
		w_tmp = self.weights # Get layer weights 
		b_tmp = self.bias # Get layer biases 
		# Calculate input weighted sum 
		z_tmp =  np.dot(w_tmp , input) + b_tmp 
		# Calculate the layer activations 
		a_tmp = self.activation(z_tmp) 
		# Store activations in cache (required for backprop)
		self.cache['A'] = a_tmp 
		# Store weighted sum in cache (required for backprop)
		self.cache['Z'] = z_tmp 
		# Store input (viz activation of prev layer) (required for backprop)
		self.cache['inputs'] = input 
		del w_tmp, b_tmp, z_tmp
		
	def backward(self,rhs_feed,batch_size):
		'''
		Inputs 
		rhs_feed ::: numpy.array (Next_Layer_Dim,Batch_Size) ::: right hand 
		side feed : DLoss/DA 
		batch_size ::: int :: The batch size 
		Outputs 
		dal_tmp ::: numpy.array (Layer_Dim,Batch_Size) ::: 
		'''
		# Init dal_tmp
		dal_tmp = rhs_feed
		#
		zl_tmp = self.cache['Z'] # get weigthed wum of the layer 
		alm1_tmp = self.cache['inputs'] # get activation of prev layer 
		wl_tmp = self.weights # get model weights 
		#
		dzl_tmp = dal_tmp * self.dactivation(zl_tmp)
		# Calculate DLoss/DW
		dwl_tmp = (1/batch_size) * np.dot(dzl_tmp , np.transpose(alm1_tmp))
		# Calculate DLoss/DB 
		dbl_tmp = (1/batch_size) * np.sum(dzl_tmp, axis = 1,keepdims = True)
		# Store weights and bias derivatives 
		self.dweights_ = dwl_tmp
		self.dbias_ = dbl_tmp
		# Update dal_tmp, for the next step (viz previous layer) of backprop
		dal_tmp = np.dot(np.transpose(wl_tmp) , dzl_tmp)
		del zl_tmp,alm1_tmp,wl_tmp,dzl_tmp,dwl_tmp,dbl_tmp
		return dal_tmp	 
