import sys as sys 
sys.path.append('../')
from sources.gaussian_mixture_numerics import * 	

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
	