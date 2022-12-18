from NN_numerics import * 

print('hi there')

layer = Dense(2,input_units = 3)
layer2 = Dense(3,input_units = 2)

W = np.array([[1,2,3],[2,4,6]])
B = np.array([[4],[0]])
layer.bias = B 
layer.weights = W

W = np.array([[1,2],[2,4],[3,6]])
B = np.array([[4],[0],[1]])
layer2.bias = B 
layer2.weights = W

input = np.array([[1],[1],[1]])
print(input)
print(np.shape(input))

layer.forward(input)
res1 = layer.cache['A']
print(res1)

layer2.forward(res1)
res2 = layer2.cache['A']
print(res2)

model = Model([layer,layer2],loss = 'MSE')
res3 = model.predict(input)
print(res3)

#
input = np.array([[1,1],[1,1],[1,1]])
print(input)
print(np.shape(input))

layer.forward(input)
res1 = layer.cache['A']
print(res1)

layer2.forward(res1)
res2 = layer2.cache['A']
print(res2)

model = Model([layer,layer2],loss = 'MSE')
res3 = model.predict(input)
print(res3)
