import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def general_gaussian_probability(x_vec,mu_vec,sigma_mat): 
	'''
	Calculate probability from a Gaussian distribution law
	arguments : 
	x_vec ::: array (n_features) ::: input variables 
	mu_vec ::: array(n_features) ::: averaged vector 
	sigma_mat ::: array (n_features,n_features) ::: covariance matrix 
	returns : 
	res ::: float ::: probability of observing x_vec
	'''
	arr_tmp = x_vec - mu_vec
	inv_sig = np.linalg.inv(sigma_mat)
	det_sig = np.linalg.det(sigma_mat)
	dim = len(x_vec)
	
	dot1 = np.matmul(inv_sig,arr_tmp).T
	dot2 = np.matmul(dot1,arr_tmp)
	dot2 = -0.5*dot2
	res = 1/((2*np.pi)**(dim/2) * det_sig**(0.5))  * np.exp(dot2)
	
	return res
	

class GaussianMixture : 
	'''
	Guassian Mixture model 
	'''
	def __init__(self,n_class):
		'''
		Gaussian Mixture init 
		arguments : 
		n_class ::: int ::: number of classes/cluster 
		'''
		self.sigma_list = None 
		self.mu_list = None 
		self.n_classes = n_class
		self.class_proportions = None
		
	def initialise_attributes(self,x):
		'''
		Initialise the gaussian distributions, given the data
		arguments :
		x ::: array (n_features, n_samples) ::: 
		updates : 
		self.sigma_list ::: list (self.n_classes) of float ::: averages 
		self.mu_list ::: list (self.n_classes) of float ::: standard deviations 
		'''
		# Split the input array in n_classes
		split_list = np.array_split(x,self.n_classes)
		# Update the gaussian distributions, given the splits 
		self.update(split_list,np.size(x,0))
		
	def update(self,split_list,data_size): 
		'''
		Update the Gaussian distributions, given data split
		arguments 
		split_list ::: list (self.n_classes) of arrays (split_size,n_features) ::: list 
		of arrays. Each array is made of observations belonging (or expected to belong) to 
		each class. 
		data_size ::: int :: total number of observations among the dataset 
		updates :
		self.sigma_list ::: list (self.n_classes) of float ::: averages 
		self.mu_list ::: list (self.n_classes) of float ::: standard deviations 
		self.class_proportions ::: list (self.n_classes) of float ::: proportion of data 
		associated with each class. 
		'''
		mu_list = []
		sigma_list = []
		proportion_list = []
		# loop over data split
		for split in split_list : 
			# calculate covariance matrix 
			sigma = np.cov(split.T)
			# calculate mean vector 
			mu= np.mean(split,axis = 0)
			# calculate proportion 
			proportion = np.size(split,0)/data_size
			# append to lists 
			sigma_list.append(sigma)
			mu_list.append(mu)
			proportion_list.append(proportion)
		# update self 
		self.mu_list = mu_list
		self.sigma_list = sigma_list
		self.class_proportions = proportion_list
		
	def utils_disp_shapes(self): 
		'''
		Display Gaussian mixture. 
		'''
		# loop over all classes 
		for i in range(self.n_classes):
			print('Class '+ str(i))
			print('mu ::: ', np.shape(self.mu_list[i]))
			print('sigma: ', np.shape(self.sigma_list[i]))
			
	def compute_law(self,x_array,mu, sigma): 
		'''
		Calculate probabilities for several observations 
		arguments :
		x_array ::: array (n_samples, n_features) ::: model input variables 
		mu ::: arrays (n_features) ::: means vector 
		sigma ::: list (self.n_classes) of arrays (n_features, n_features) ::: coavariance matrix  
		returns :
		probabilities ::: array (n_samples) ::: probability of observing x_array, given a Gaussian law 
		of mean mu and covariance matrix sigma 
		'''
		return np.array([general_gaussian_probability(x_array[i,:],mu,sigma) for i in range(np.size(x_array,0))])
			
	def predict_proba(self,input_data): 
		'''
		Calculate probabilities associated to each observation from the input_data 
		arguments : 
		input_data ::: array (n_samples,n_features) ::: Model input data 
		returns :
		probabilities ::: array (n_samples,n_classe) ::: probabilities 
		'''
		# init probabilities matrix 
		probabilities = np.zeros((np.size(input_data,0),self.n_classes))
		# classes loop 
		for k in range(self.n_classes) : 
			# calculate prbability from a single Gaussian law 
			arr_tmp = self.class_proportions[k]*self.compute_law(input_data,self.mu_list[k],self.sigma_list[k])
			probabilities[:,k] = arr_tmp
			# Scale the probability, with the sum of probabilities of all gaussians 
			probabilities[:,k] /= np.sum([self.class_proportions[j]*self.compute_law(input_data,
																			self.mu_list[j],
																			self.sigma_list[j]) for j in range(self.n_classes)])
		return probabilities
	
	def predict(self,input_data) : 
		'''
		Make a prediction, given input data 
		arguments : 
		input_data ::: array (n_samples, n_features) ::: Model input data 
		returns : 
		classes ::: array (n_samples) ::: predicted classes 
		'''
		probabilities = self.predict_proba(input_data)
		classes = np.argmax(probabilities,axis = 1)
		return classes
		
	def predict_split(self,input_data):
		'''
		Split input_data, according to the probabilities of brlonging to each classes of the model 
		argument :
		input_data ::: array (n_samples, n_features) ::: Model input data 
		returns ::: 
		split_list ::: list (self.n_classes) of arrays (split_size,n_features) ::: list 
		of arrays. 
		'''
		classes = self.predict(input_data)
		split_list = [input_data[np.argwhere(classes == i).squeeze(),:] for i in range(self.n_classes) ]
		return split_list
		 
	def fit(self,input_data,max_iter = 100): 
		'''
		Train Gaussians Mixture model given input data 
		arguments 
		input_data ::: array (n_samples, n_features) ::: Model input data 
		max_iter ::: int ::: maximum number of iteration for training the Model 
		updates 
		self.sigma_list ::: list (self.n_classes) of float ::: averages 
		self.mu_list ::: list (self.n_classes) of float ::: standard deviations 
		self.class_proportions ::: list (self.n_classes) of float ::: proportion of data 
		associated with each class. 
		'''
		# initialize Gaussian laws 
		self.initialise_attributes(input_data)
		# get problem dimensions 
		data_size = np.size(input_data,0)
		data_dim = np.size(input_data,1)
		# init stop criterion 
		stop_criterion = False 
		# inut arrays that will contain the Gaussian laws of the former step 
		# these are used for stoping the model training 
		prev_mu = [np.zeros(data_dim) for i in range(self.n_classes)]
		prev_sigma = [np.zeros((data_dim,data_dim)) for i in range(self.n_classes)]
		n = 0
		# training loop 
		while not stop_criterion and (n < max_iter) : 
			print(n)
			# split data according to probabilities 
			split_list = self.predict_split(input_data)
			# update Gaussian laws, from the split 
			self.update(split_list,data_size)
			# Compare current Gaussian laws with the former ones 
			# if Gaussian laws have not changed then stop_criterion 
			# is set to True 
			stop_criterion = np.all(np.asarray(prev_mu) == np.asarray(self.mu_list)) and np.all(np.asarray(prev_sigma) == np.asarray(self.sigma_list))
			print(stop_criterion)
			# Update prev_mu and prev_sigma, for future step
			prev_mu = self.mu_list
			prev_sigma = self.sigma_list
			#
			n = n + 1
			
if __name__ == '__main__':
	# Set the mean and covariance
	mean1 = [0, 0]
	mean2 = [2, 0]
	mean3 = [8,10]
	cov1 = [[1, .7], [.7, 1]]
	cov2 = [[.5, .4], [.4, .5]]
	cov3 = [[2, .4], [.4, .2]]
	
	# Generate data from the mean and covariance
	data1 = np.random.multivariate_normal(mean1, cov1, size=1000)
	data2 = np.random.multivariate_normal(mean2, cov2, size=1000)
	data3 = np.random.multivariate_normal(mean3, cov3, size=1000)
	data_vec = np.concatenate((data1,data2,data3),axis = 0)
	ind_list = np.arange(np.size(data_vec,0))
	np.random.shuffle(ind_list)
	print(ind_list)
	data_vec = data_vec[ind_list,:]
	#data_vec = np.array([[2,4],[8,3],[1.5,1.5],[1.5,2.5],[1.4,3],[7.5,3.2],[8,4],[7.2,3.4]])
	
	gm = GaussianMixture(3)
	gm.initialise_attributes(data_vec)
	
	
	
	x = np.linspace(0,10,100)
	xv, yv = np.meshgrid(x,x)
	print(xv.shape,yv.shape)
	proba = np.zeros((100,100))
	print(proba.shape)
	for i in range(100): 
		for j in range(100):
			proba[i,j] = general_gaussian_probability(np.array([xv[i,j],yv[i,j]]), gm.mu_list[0], gm.sigma_list[0] )
	plt.pcolormesh(xv,yv,proba)
	plt.scatter(data_vec[:,0],data_vec[:,1],c=gm.predict(data_vec))
	plt.show()
	
	gm.fit(data_vec,max_iter = 100)
	
	x = np.linspace(0,10,100)
	xv, yv = np.meshgrid(x,x)
	print(xv.shape,yv.shape)
	proba = np.zeros((100,100))
	print(proba.shape)
	for i in range(100): 
		for j in range(100):
			proba[i,j] = general_gaussian_probability(np.array([xv[i,j],yv[i,j]]), gm.mu_list[0], gm.sigma_list[0] )
	plt.pcolormesh(xv,yv,proba)
	plt.scatter(data_vec[:,0],data_vec[:,1],c=gm.predict(data_vec))
	plt.show()
	
	


	
	
	

	

	
	
	
