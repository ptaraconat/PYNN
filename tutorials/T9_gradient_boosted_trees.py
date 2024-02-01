import sys as sys 
sys.path.append('../')
from sources.random_forest import GradientBoostedTrees
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.feature_selection import r_regression

# Data 
Ndata = 100
X = np.linspace(0,3,Ndata) #np.ones((Ninput,5))
X = np.expand_dims(X,1)
Y = np.power(X,2)#+np.random.normal(loc=0.1, scale=4.0)

# model 
model = GradientBoostedTrees(n_trees = 100,min_samples_splits = 10, learning_rate= 0.1)
res = model.fit(X,Y)
Y_hat = model.predict(X)

print(Y_hat)
print(r_regression(Y,Y_hat)[0]**2.)
