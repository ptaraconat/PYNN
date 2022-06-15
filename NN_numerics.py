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

class MLP_Model():
	def __init__(self,Lay = [4,3,2,1],learningrate = 0.001,activations = [],dactivations = [], activation = linear,dactivation = dlinear,Loss = MSE,DLoss = DMSE):
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
		i = 1
		while i < Nlay : 
			nl = Lay[i]
			nlm1 = Lay[i-1]
			self.Learned['W'+str(i)] =  np.zeros((nl,nlm1)) + np.random.normal(loc=0.0, scale=1.0)
			self.Learned['B'+str(i)] = np.zeros((nl,1)) + np.random.normal(loc=0.0, scale=1.0)
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
		l = 1 
		while l < self.Nlay:
			self.Learned['W'+str(l)] = self.Learned['W'+str(l)] - self.LearningRate*self.Cache['dW'+str(l)]
			self.Learned['B'+str(l)] = self.Learned['B'+str(l)] - self.LearningRate*self.Cache['dB'+str(l)]
			l = l + 1 
	
	def Fit(self,X,Yhat,Nmax=1000):
		i = 0 
		err = []
		while (i < Nmax) :
			Y = self.Predict(X)
			loss = self.CalcCurrentLoss(Y,Yhat)
			self.BackProp(Y,Yhat)
			self.Update()
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
	model = MLP_Model(Lay = Lay,learningrate= 0.01,activations = act,dactivations = dact)
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
	
	
	


			
