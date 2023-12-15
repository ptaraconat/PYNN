import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def general_gaussian_probability(x_vec,mu_vec,sigma_mat): 
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
	
	def __init__(self,n_class):
		self.sigma_list = None 
		self.mu_list = None 
		self.n_classes = n_class
		self.class_proportions = None
		
	def initialise_attributes(self,x):
		
		split_list = np.array_split(x,self.n_classes)
		self.update(split_list,np.size(x,0))
		
	def update(self,split_list,data_size): 
		mu_list = []
		sigma_list = []
		proportion_list = []
		for split in split_list : 
			sigma = np.cov(split.T)
			mu= np.mean(split,axis = 0)
			proportion = np.size(split,0)/data_size
			sigma_list.append(sigma)
			mu_list.append(mu)
			proportion_list.append(proportion)
		self.mu_list = mu_list
		self.sigma_list = sigma_list
		self.class_proportions = proportion_list
		
	def utils_disp_shapes(self): 
		for i in range(self.n_classes):
			print('Class '+ str(i))
			print('mu ::: ', np.shape(self.mu_list[i]))
			print('sigma: ', np.shape(self.sigma_list[i]))
			
	def compute_law(self,x_array,mu, sigma): 
		return np.array([general_gaussian_probability(x_array[i,:],mu,sigma) for i in range(np.size(x_array,0))])
			
	def predict_proba(self,input_data): 
		probabilities = np.zeros((np.size(input_data,0),self.n_classes))
		for k in range(self.n_classes) : 
			arr_tmp = self.class_proportions[k]*self.compute_law(input_data,self.mu_list[k],self.sigma_list[k])
			probabilities[:,k] = arr_tmp
			probabilities[:,k] /= np.sum([self.class_proportions[j]*self.compute_law(input_data,self.mu_list[j],self.sigma_list[j]) for j in range(self.n_classes)])
		#denominator = np.sum(self.class_proportions*probabilities,axis = 1)
		#probabilities = np.divide(probabilities,np.expand_dims(denominator,axis = 1))
		return probabilities
	
	def predict(self,input_data) : 
		probabilities = self.predict_proba(input_data)
		classes = np.argmax(probabilities,axis = 1)
		return classes
		
	def predict_split(self,input_data):
		 classes = self.predict(input_data)
		 split_list = [input_data[np.argwhere(classes == i).squeeze(),:] for i in range(self.n_classes) ]
		 return split_list
		 
	def fit(self,input_data,max_iter = 100): 
		self.initialise_attributes(input_data)
		data_size = np.size(input_data,0)
		data_dim = np.size(input_data,1)
		stop_criterion = False 
		prev_mu = [np.zeros(data_dim) for i in range(self.n_classes)]
		prev_sigma = [np.zeros((data_dim,data_dim)) for i in range(self.n_classes)]
		n = 0
		while not stop_criterion and (n < max_iter) : 
			print(n)
			split_list = self.predict_split(input_data)
			self.update(split_list,data_size)
			
			stop_criterion = np.all(np.asarray(prev_mu) == np.asarray(self.mu_list)) and np.all(np.asarray(prev_sigma) == np.asarray(self.sigma_list))
			print(stop_criterion)
			prev_mu = self.mu_list
			prev_sigma = self.sigma_list
			
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
	
	


	
	
	

	

	
	
	
