import sys as sys 
sys.path.append('../../')
from PYNN.sources.neural_net import Dense, Model, Adam, SGD
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.feature_selection import r_regression

# Data 
Ndata = 100
X = np.linspace(0,3,Ndata) #np.ones((Ninput,5))
X = np.expand_dims(X,0)
Yhat = np.power(X,2)#+np.random.normal(loc=0.1, scale=4.0)
# Define Model
layer1 = Dense(4,activation ='sigmoid',input_units = 1)
layer2 = Dense(2,activation ='sigmoid',input_units = 4)
layer3 = Dense(1,activation ='linear',input_units = 2) 
model = Model(layers = [layer1,layer2,layer3],loss = 'MSE')
model.summary()
model.disp_learnable()
# Learning
opt1 = Adam(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8)
opt2 = SGD(learning_rate = 0.01)
err = model.fit(X,Yhat,optimizer = opt1, epochs = 10000,batch_size = 30)
model.disp_learnable()
Y = model.predict(X)
#Test Model 
plt.plot(X[0],Yhat[0],'bo',markersize = 8)
plt.plot(X[0],Y[0],'k-',linewidth = 3)
#plt.savefig('comp.png',format = 'png')
plt.show()
plt.close()
# Plot Error curve 
plt.plot(err) 
plt.show()
plt.close()
#plt.savefig('err.png',format = 'png')
plt.close()
print(Y)

print(r_regression(Y,-Y_hat))