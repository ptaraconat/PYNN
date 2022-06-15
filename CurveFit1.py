from NN_numerics import * 

# Model Settings 
Ninput = 1
Noutput = 1
Lay = [Ninput,4,2,Noutput]
act = [sigmoid,sigmoid,linear]
dact = [dsigmoid,dsigmoid,dlinear]
model = MLP_Model(Lay = Lay,learningrate= 0.01,activations = act,dactivations = dact)
model.Summary()
# Data 
Ndata = 100
X = np.linspace(0,3,Ndata) #np.ones((Ninput,5))
X = np.expand_dims(X,0)
Yhat = np.power(X,2)#+np.random.normal(loc=0.1, scale=4.0)
# Learning
err = model.Fit(X,Yhat,Nmax = 10000)
Y = model.Predict(X)
# Plot Error curve 
plt.plot(err) 
plt.show()
plt.close()
#Test Model 
plt.plot(X[0],Yhat[0],'bo',markersize = 8)
plt.plot(X[0],Y[0],'k-',linewidth = 3)
plt.show()
plt.close()
