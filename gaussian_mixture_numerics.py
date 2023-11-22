import numpy as np 
import pandas as pd 

def general_gaussian_probability(x_vec,mu_vec,sigma_mat): 
	inv_sigma_mat = np.linalg.inv(sigma_mat)
	det_sigma_mat = np.linalg.det(sigma_mat)
	return np.dot(np.dot(inv_sigma_mat,x_vec-mu_vec),x_vec-mu_vec)
	

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
	mu1 = np.array([[2,4]])
	sigma1 = np.array([[2,0],[0,2]])
	mu2 = np.array([[4,8]])
	sigma2 = np.array([[1,0],[0,1]])
	mu3 = np.array([[8,16]])
	sigma3 = np.array([[1,0],[0,1]])
	gm.n_classes = 3
	gm.mu_list = [mu1,mu2,mu3]
	gm.sigma_list = [sigma1, sigma2, sigma3]
	gm.utils_disp_shapes()
	
	data_vec = np.array([[1,4],[1,3],[4,8]])
	
	arr_tmp = data_vec - mu1
	inv_sig_1 = np.linalg.inv(sigma1)
	print(inv_sig_1)
	print(arr_tmp)
	dot1 = np.dot(inv_sig_1,arr_tmp.T)
	print(dot1)
	print(arr_tmp.shape,dot1.shape)
	dot2 = np.dot(dot1,arr_tmp)
	print(dot2)
	
	
	
