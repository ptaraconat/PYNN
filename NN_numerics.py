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
		while l >= 0: 
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
		l = 0
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
		self.algorithm = 'Adam'
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
		
	def summary(self):
		print('##### Model Summary #####')
		for i in range(len(self.layers)):
			print('###########################')
			print('Layer Number '+str(i+1))
			print('Weights ::: ',np.shape(self.layers[i].weights))
			print('Biases  ::: ',np.shape(self.layers[i].bias))
			
	def disp_learnable(self):
		print('##### Model Summary #####')
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
		self.units = units
		self.cache = dict()
		self.cache['Sdb'] = np.zeros((self.units,1))
		self.cache['Vdb'] = np.zeros((self.units,1))
		self.bias = np.zeros((units,1)) + np.random.normal(loc=0.0, scale=1.0,size = (units,1))
		if activation == 'linear':
			self.activation = linear
			self.dactivation = dlinear
		if activation == 'sigmoid':
			self.activation = sigmoid
			self.dactivation = dsigmoid
		if input_units != None : 
			self.input_units = input_units
			self.weights = np.zeros((self.units,self.input_units)) + np.random.normal(loc=0.0, scale=1.0,size = (self.units,self.input_units))
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
			self.cache['Sdb'] = beta2*self.cache['Sdb'] + (1-beta2)*np.square(self.cache['dB'])
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
				
if __name__ == "__main__":
	# Data 
	X = np.ones((4,4))
	Yhat = np.ones((2,4))*2#np.array([[2],[2]])
	#########
	#layer1 = Dense(4,activation ='linear',input_units = 4)
	layer1 = Dense(3,activation ='sigmoid',input_units = 4)
	layer2 = Dense(2,activation ='sigmoid',input_units = 3)
	layer3 = Dense(2,activation ='linear',input_units = 2)
	model2 = Model(layers = [layer1,layer2,layer3],loss = 'MSE')
	err2 = model2.Fit(X,Yhat,Nmax = 1000)
	plt.plot(err2)
	plt.show()
	
	print(model2.predict(X))
	print(Yhat)	
	
	


			
