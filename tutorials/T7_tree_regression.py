import sys as sys 
sys.path.append('../../')
from PYNN.sources.tree_models import TreeRegressor
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.feature_selection import r_regression

# Data 
Ndata = 100
X = np.linspace(0,3,Ndata) #np.ones((Ninput,5))
X = np.expand_dims(X,1)
Y = np.power(X,2)#+np.random.normal(loc=0.1, scale=4.0)

# model 
model = TreeRegressor(min_sample_split = 5)
model.fit(X,Y)
Y_hat = model.predict(X)

print(Y_hat)
print(r_regression(Y,Y_hat)[0]**2.)