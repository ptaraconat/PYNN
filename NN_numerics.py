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
	
def MSE(Y,Yhat):
	res = np.power(Yhat - Y,2)
	res = np.sum(res,axis = 0,keepdims = True) 
	N = np.size(Yhat,0)
	res = res/N
	return res

def DMSE(Y,Yhat):
	N = np.size(Yhat,0)
	return (2/N)*(Y-Yhat)

class Model():
	def __init__(self,layers = [],loss = 'MSE'):
		self.layers = layers 
		if loss == 'MSE' : 
			self.loss = MSE
			self.dloss = DMSE
	
	def predict(self,inputs): 
		input_tmp = inputs
		for layer in self.layers : 
			layer.forward(input_tmp)
			input_tmp = layer.cache['A']
		return input_tmp
	
	def backprop(self,Y,Yhat):
		batch_size = np.size(Y,1)
		rhs_feed = self.dloss(Y,Yhat)
		l = len(self.layers) - 1
		while l > 0: 
			rhs_feed = self.layers[l].backward(rhs_feed,batch_size)
			l = l - 1 
	
	def update(self):
		#to be reworked 
		beta1 = self.Beta1
		beta2 = self.Beta2
		ite = self.Ite
		epsilon = self.Epsilon 
		learning_rate = self.learning_rate
		algo = self.algorithm
		#
		l = 1 
		while l < len(self.layers):
			self.layers[l].update(learning_rate,beta1,beta2,epsilon,ite,Algorithm = algo)
			l = l + 1
		
	def calc_loss(self,Y,Yhat): 
		Nex = np.size(Y,1)
		err = self.loss(Y,Yhat)
		loss = np.sum(err)/Nex
		return loss
		
	def Fit(self,X,Yhat,Nmax=1000):
		i = 0 
		err = []
		self.Ite = 1
		self.Beta1 = 0.9
		self.Beta2 = 0.999
		self.Epsilon = 10e-8
		self.learning_rate = 0.01
		self.algorithm = 'SGD'
		while (i < Nmax) :
			Y = self.predict(X)
			loss = self.calc_loss(Y,Yhat)
			self.backprop(Y,Yhat)
			self.update()
			self.Ite = self.Ite + 1 
			#
			err = err + [loss]
			del Y, loss
			i = i + 1 
		return err
		

class Layer():
	def __init__(self):
		self.Name = None

class Dense(Layer):
	def __init__(self,units,activation ='linear',input_units = None): 
		self.units = units
		self.cache = dict()
		self.cache['Sdb'] = np.zeros((self.units,1))
		self.cache['Vdb'] = np.zeros((self.units,1))
		self.bias = np.zeros((units,1))
		if activation == 'linear':
			self.activation = linear
			self.dactivation = dlinear
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.dactivation = dsigmoid
		if input_units != None : 
			self.input_units = input_units
			self.weights = np.zeros((self.units,self.input_units))
			self.cache['Sdw'] = np.zeros((self.units,self.input_units))
			self.cache['Vdw'] = np.zeros((self.units,self.input_units))
			
	def forward(self,input): 
		W_tmp = self.weights#self.Learned['W'+str(i)]
		B_tmp = self.bias #self.Learned['B'+str(i)]
		z_tmp =  np.dot(W_tmp , input) + B_tmp
		a_tmp = self.activation(z_tmp)#self.Activations[i-1](z_tmp)
		self.cache['A'] = a_tmp
		self.cache['Z'] = z_tmp
		self.cache['inputs'] = input
		del W_tmp, B_tmp, z_tmp
		
	def backward(self,rhs_feed,batch_size):
		#
		dAl_tmp = rhs_feed#self.DCostFunc(Y,Yhat)
		#
		Zl_tmp = self.cache['Z']
		Alm1_tmp = self.cache['inputs']#prev_lay.cache['A']
		Wl_tmp = self.weights
		M = batch_size #np.size(Yhat,1)
		#
		#dZl_tmp = dAl_tmp * self.DActivation(Zl_tmp) 
		dZl_tmp = dAl_tmp * self.dactivation(Zl_tmp)#self.DActivations[l-1](Zl_tmp) ### minus one because stored in list 
		dWl_tmp = (1/M) * np.dot(dZl_tmp , np.transpose(Alm1_tmp))
		dbl_tmp = (1/M) * np.sum(dZl_tmp, axis = 1,keepdims = True)
		self.cache['dW'] = dWl_tmp
		self.cache['dB'] = dbl_tmp
		#
		dAl_tmp = np.dot(np.transpose(Wl_tmp) , dZl_tmp)
		del Zl_tmp,Alm1_tmp,Wl_tmp,M,dZl_tmp,dWl_tmp,dbl_tmp
		return dAl_tmp

	def update(self,learning_rate,beta1,beta2,epsilon,ite,Algorithm = 'SGD'):
		if Algorithm == 'SGD':
			self.weights = self.weights - learning_rate*self.cache['dW']
			self.bias = self.bias - learning_rate*self.cache['dB']
				
		if Algorithm == 'Adam' :  
			# Update Sd
			self.cache['Sdw'] = beta2*self.cache['Sdw'] + (1-beta2)*np.square(self.cache['dW'])
			self.cache['Sdb'] = beta2*self.cache['Sdb'] + (1-beta2)*np.square(self.Cache['dB'])
			# Update Vd
			self.cache['Vdw'] = beta1*self.cache['Vdw'] + (1-beta1)*self.cache['dW'] 
			self.cache['Vdb'] = beta1*self.cache['Vdb'] + (1-beta1)*self.cache['dB'] 
			# Correct Vd and Sd 
			Vdwc_tmp = self.cache['Vdw']/(1-(beta1**ite))
			Vdbc_tmp = self.cache['Vdb']/(1-(beta1**ite))
			Sdwc_tmp = self.cache['Sdw']/(1-(beta2**ite))
			Sdbc_tmp = self.cache['Sdb']/(1-(beta2**ite))
			#Update Model weights, using Vd and Sd: to be completed 
			self.weights =  self.weights - learning_rate * np.divide(Vdwc_tmp,(np.sqrt(Sdwc_tmp)+epsilon))
			self.bias = self.bias - learning_rate*np.divide(Vdbc_tmp,(np.sqrt(Sdbc_tmp)+epsilon))
			 
class Optimizer():
	
	def __init__(self,type = None):
		self.type = None

class Adam(Optimizer):
	
	def __init__(self,learning_rate,beta1, beta2, epsilon):
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.learning_rate = learning_rate

class SGD(Optimizer):
	def __init__(self,learning_rate):
		self.learning_rate = learning_rate

class MLP_Model():
	def __init__(self,Lay = [4,3,2,1],learningrate = 0.001,activations = [],dactivations = [], activation = linear,dactivation = dlinear,Loss = MSE,DLoss = DMSE,algorithm = 'SGD'):
		Nlay = len(Lay)
		self.Nlay = Nlay
		self.Lay = Lay
		self.Learned = dict()
		#self.Activation = activation
		#self.DActivation = dactivation
		self.Activations = activations
		self.DActivations = dactivations
		if (activations == []) or (dactivations == []):
			i = 0 
			while i < Nlay-1 : 
				self.Activations = self.Activations + [activation]
				self.DActivations = self.DActivations + [dactivation]
				i = i + 1
		self.Cache = dict()
		self.CostFunc = Loss
		self.DCostFunc = DLoss
		self.LearningRate = learningrate
		self.Algorithm = algorithm
		i = 1
		while i < Nlay : 
			nl = Lay[i]
			nlm1 = Lay[i-1]
			self.Learned['W'+str(i)] =  np.zeros((nl,nlm1)) + np.random.normal(loc=0.0, scale=1.0)
			self.Learned['B'+str(i)] = np.zeros((nl,1)) + np.random.normal(loc=0.0, scale=1.0)
			if self.Algorithm == 'RMSProp' or self.Algorithm == 'Adam': 
				self.Cache['Sdw'+str(i)] = np.zeros((nl,nlm1))
				self.Cache['Sdb'+str(i)] = np.zeros((nl,1))
			if self.Algorithm == 'Momentum' or self.Algorithm == 'Adam':
				self.Cache['Vdw'+str(i)] = np.zeros((nl,nlm1))
				self.Cache['Vdb'+str(i)] = np.zeros((nl,1))	
			i = i + 1 
		
	def Summary(self):
		print('##### Model Summary #####')
		for i in range(self.Nlay-1):
			print('###########################')
			print('Layer Number '+str(i+1))
			print('Weights ::: ',np.shape(self.Learned['W'+str(i+1)]))
			print('Biases  ::: ',np.shape(self.Learned['B'+str(i+1)]))
	
	def Predict(self,X):
		i = 1
		a_tmp = X
		self.Cache['A'+str(0)] = a_tmp
		while i < self.Nlay:
			W_tmp = self.Learned['W'+str(i)]
			B_tmp = self.Learned['B'+str(i)]
			z_tmp =  np.dot(W_tmp , a_tmp) + B_tmp
			#a_tmp = self.Activation(z_tmp)
			a_tmp = self.Activations[i-1](z_tmp)
			self.Cache['A'+str(i)] = a_tmp
			self.Cache['Z'+str(i)] = z_tmp
			del W_tmp, B_tmp, z_tmp
			i = i + 1 
		return a_tmp
	
	def CalcCurrentLoss(self,Y,Yhat):
		Nex = np.size(Y,1)
		err = self.CostFunc(Y,Yhat)
		return np.sum(err)/Nex
		
	def BackProp(self,Y,Yhat):
		dAl_tmp = self.DCostFunc(Y,Yhat)
		l = self.Nlay - 1
		while l > 0: 
			#
			Zl_tmp = self.Cache['Z'+str(l)]
			Alm1_tmp = self.Cache['A'+str(l-1)]
			Wl_tmp = self.Learned['W'+str(l)]
			M = np.size(Yhat,1)
			#
			#dZl_tmp = dAl_tmp * self.DActivation(Zl_tmp) 
			dZl_tmp = dAl_tmp * self.DActivations[l-1](Zl_tmp) ### minus one because stored in list 
			dWl_tmp = (1/M) * np.dot(dZl_tmp , np.transpose(Alm1_tmp))
			dbl_tmp = (1/M) * np.sum(dZl_tmp, axis = 1,keepdims = True)
			self.Cache['dW'+str(l)] = dWl_tmp
			self.Cache['dB'+str(l)] = dbl_tmp
			#
			dAl_tmp = np.dot(np.transpose(Wl_tmp) , dZl_tmp)
			del Zl_tmp,Alm1_tmp,Wl_tmp,M,dZl_tmp,dWl_tmp,dbl_tmp
			# 
			l = l - 1 
			
	def Update(self):
		#to be reworked 
		beta1 = self.Beta1
		beta2 = self.Beta2
		ite = self.Ite
		epsilon = self.Epsilon 
		#
		l = 1 
		while l < self.Nlay:
			if self.Algorithm == 'SGD':
				self.Learned['W'+str(l)] = self.Learned['W'+str(l)] - self.LearningRate*self.Cache['dW'+str(l)]
				self.Learned['B'+str(l)] = self.Learned['B'+str(l)] - self.LearningRate*self.Cache['dB'+str(l)]
				
			if self.Algorithm == 'Adam' :  
				# Update Sd
				self.Cache['Sdw'+str(l)] = beta2*self.Cache['Sdw'+str(l)] + (1-beta2)*np.square(self.Cache['dW'+str(l)])
				self.Cache['Sdb'+str(l)] = beta2*self.Cache['Sdb'+str(l)] + (1-beta2)*np.square(self.Cache['dB'+str(l)])
				# Update Vd
				self.Cache['Vdw'+str(l)] = beta1*self.Cache['Vdw'+str(l)] + (1-beta1)*self.Cache['dW'+str(l)] 
				self.Cache['Vdb'+str(l)] = beta1*self.Cache['Vdb'+str(l)] + (1-beta1)*self.Cache['dB'+str(l)] 
				# Correct Vd and Sd 
				Vdwc_tmp = self.Cache['Vdw'+str(l)]/(1-(beta1**ite))
				Vdbc_tmp = self.Cache['Vdb'+str(l)]/(1-(beta1**ite))
				Sdwc_tmp = self.Cache['Sdw'+str(l)]/(1-(beta2**ite))
				Sdbc_tmp = self.Cache['Sdb'+str(l)]/(1-(beta2**ite))
				#Update Model weights, using Vd and Sd: to be completed 
				self.Learned['W'+str(l)] =  self.Learned['W'+str(l)] - self.LearningRate * np.divide(Vdwc_tmp,(np.sqrt(Sdwc_tmp)+epsilon))
				self.Learned['B'+str(l)] = self.Learned['B'+str(l)] - self.LearningRate*np.divide(Vdbc_tmp,(np.sqrt(Sdbc_tmp)+epsilon))
			l = l + 1 
	
	def Fit(self,X,Yhat,Nmax=1000):
		i = 0 
		err = []
		self.Ite = 1
		self.Beta1 = 0.9
		self.Beta2 = 0.999
		self.Epsilon = 10e-8
		while (i < Nmax) :
			Y = self.Predict(X)
			loss = self.CalcCurrentLoss(Y,Yhat)
			self.BackProp(Y,Yhat)
			self.Update()
			self.Ite = self.Ite + 1 
			#
			err = err + [loss]
			del Y, loss
			i = i + 1 
		return err
				
if __name__ == "__main__":
	# Model Settings 
	Lay = [4,3,2,2]
	act = [sigmoid,sigmoid,linear]
	dact = [dsigmoid,dsigmoid,dlinear]
	model = MLP_Model(Lay = Lay,learningrate= 0.01,activations = act,dactivations = dact,algorithm='Adam')
	model.Summary()
	# Data 
	X = np.ones((4,4))
	Yhat = np.ones((2,4))*2#np.array([[2],[2]])
	# Learn and Test 
	Y = model.Predict(X)
	print(Y)
	err = model.Fit(X,Yhat,Nmax = 1000)
	Y = model.Predict(X)
	print(Y)
	print(Yhat)
	# Plot Error curve 
	plt.plot(err)
	plt.show()
	plt.close()
	#########
	#layer1 = Dense(4,activation ='linear',input_units = 4)
	layer1 = Dense(3,activation ='sigmoid',input_units = 4)
	layer2 = Dense(2,activation ='sigmoid',input_units = 3)
	layer3 = Dense(2,activation ='linear',input_units = 2)
	model2 = Model(layers = [layer1,layer2,layer3],loss = 'MSE')
	err2 = model2.Fit(X,Yhat,Nmax = 1000)
	plt.plot(err2)
	plt.plot(err)
	plt.show()
	
	print(model2.predict(X))
	print(Yhat)	
	
	


			
