import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def general_gaussian_probability(x_vec,mu_vec,sigma_mat): 
	arr_tmp = x_vec - mu_vec
	inv_sig = np.linalg.inv(sigma_mat)
	det_sig = np.linalg.det(sigma_mat)
	dim = len(x)
	
	dot1 = np.matmul(inv_sig,arr_tmp).T
	dot2 = np.matmul(dot1,arr_tmp)
	dot2 = -0.5*dot2
	res = 1/((2*np.pi)**(dim/2) * det_sig**(0.5))  * np.exp(dot2)
	
	return res
	

class GaussianMixture : 
	
	def __init__(self):
		self.sigma_list = None 
		self.mu_list = None 
		self.n_classes = None 
		self.class_proportions = None 
		
	def utils_disp_shapes(self): 
		for i in range(self.n_classes):
			print('Class '+ str(i))
			print('mu ::: ', np.shape(self.mu_list[i]))
			print('sigma: ', np.shape(self.sigma_list[i]))
	

if __name__ == '__main__':
	gm = GaussianMixture()
	mu1 = np.array([2,4])
	sigma1 = np.array([[2,0],[0,2]])
	mu2 = np.array([4,8])
	sigma2 = np.array([[1,0],[0,1]])
	mu3 = np.array([8,16])
	sigma3 = np.array([[1,0],[0,1]])
	gm.n_classes = 3
	gm.mu_list = [mu1,mu2,mu3]
	gm.sigma_list = [sigma1, sigma2, sigma3]
	gm.utils_disp_shapes()
	
	data_vec = np.array([[2,4],[1,3],[4,8]])
	x = data_vec[0,:]
	
	print(general_gaussian_probability(x,mu1,sigma1))
	
	x = np.linspace(0,10,100)
	xv, yv = np.meshgrid(x,x)
	print(xv.shape,yv.shape)
	proba = np.zeros((100,100))
	print(proba.shape)
	for i in range(100): 
		for j in range(100):
			proba[i,j] = general_gaussian_probability(np.array([xv[i,j],yv[i,j]]), mu1, sigma1)
	
	plt.pcolormesh(xv,yv,proba)
	plt.show()
	
	

	

	
	
	
