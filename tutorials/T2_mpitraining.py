import sys as sys 
sys.path.append('../')
from sources.neural_net import * 
from mpi4py import MPI 

def reduce_sum_func(a,b):
	return a+b

class MPIModel(Model):
    def __init__(self,layers = [],loss = 'MSE'):
        self.layers = layers 
        if loss == 'MSE' : 
            self.loss = mean_squared_error
            self.dloss = d_mean_squared_error
    def step(self,x_mb,y_mb): 
        ## Forward prop
        yhat = self.predict(x_mb)
        ## Calc Loss
        loss = self.calc_loss(yhat,y_mb)
        print('loss :' ,loss)
        ## Back prop
        self.backprop(yhat,y_mb)

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

print('worker : ',worker)
if worker == 0 :
    # Data 
    Ndata = 100
    X = np.linspace(0,3,Ndata) 
    X = np.expand_dims(X,0)
    Y = np.power(X,2)
    y_chunk = np.array_split(Y,size,axis = 1)
    x_chunk = np.array_split(X,size,axis = 1)
    
    # Define Model
    layer1 = Dense(4,activation ='sigmoid',input_units = 1)
    layer2 = Dense(1,activation ='sigmoid',input_units = 4)
    model = MPIModel(layers = [layer1,layer2],loss = 'MSE')
    # Define optimizer 
    optimizer = SGD(learning_rate= 0.01)
else : 
    x_chunk = None
    y_chunk = None 
    model = None 

X = comm.scatter(x_chunk, root = 0)
Y = comm.scatter(y_chunk, root = 0)
model = comm.bcast(model, root = 0)

model.step(X,Y)
for i in range(len(model.layers)): 
    print(i)
    global_sum_w = np.zeros(model.layers[i].dweights_.shape, dtype=float)
    comm.Reduce(model.layers[i].dweights_, global_sum_w, op=MPI.SUM, root=0)
    global_sum_b = np.zeros(model.layers[i].dbias_ .shape, dtype=float)
    comm.Reduce(model.layers[i].dbias_, global_sum_b, op=MPI.SUM, root=0)
    if worker == 0 :
        model.layers[i].dweights_ = global_sum_w
        model.layers[i].dbias_ = global_sum_b
if worker == 0 :
    model.update(optimizer)
model = comm.bcast(model, root = 0)

