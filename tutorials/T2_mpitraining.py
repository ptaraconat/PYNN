import sys as sys 
sys.path.append('../')
from sources.neural_net import * 
from mpi4py import MPI 
from math import * 

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
        ## Back prop
        self.backprop(yhat,y_mb)
        return loss

comm = MPI.COMM_WORLD
worker = comm.Get_rank()
size = comm.Get_size()

print('worker : ',worker)
# Init workers 
n_epochs = 10000
if worker == 0 :
    # hyper parameter
    batch_size_per_worker = 130
    batch_size = batch_size_per_worker * size
    learning_rate = 0.01 
    # Data 
    Ndata = 1000
    X_data = np.linspace(0,3,Ndata) 
    X_data = np.expand_dims(X_data,0)
    Y_data = np.power(X_data,2)
    n_samples = np.size(X_data,-1)
    # Define Model
    layer1 = Dense(4,activation ='sigmoid',input_units = 1)
    layer2 = Dense(1,activation ='linear',input_units = 4)
    model = MPIModel(layers = [layer1,layer2],loss = 'MSE')
    # Define optimizer 
    optimizer = SGD(learning_rate= learning_rate)
    optimizer = Adam(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8)

## Training Loop 
for epoch in range(n_epochs):
    # Shuffle and Batch data 
    if worker == 0 :
        ## Shuffle training examples 
        shuffled_indices = np.random.permutation(n_samples)
        X_data = X_data[:,shuffled_indices]
        Y_data = Y_data[:,shuffled_indices]
        ## Batch data 
        split_size = floor(n_samples/batch_size)
        x_batches = np.array_split(X_data, split_size,axis = 1)
        y_batches = np.array_split(Y_data, split_size,axis = 1)
        n_batch_ite = len(x_batches)
    else : 
        n_batch_ite = None 
    ##### Communication #######
    n_batch_ite = comm.bcast(n_batch_ite, root = 0)
    ###########################
    ## Batches loop
    batch_ite = 0
    for batch_ite in range(n_batch_ite) : 
        # Send model and data to slaves 
        if worker == 0 : 
            X_batch, Y_batch = x_batches[batch_ite], y_batches[batch_ite]
            y_chunk = np.array_split(Y_batch,size,axis = 1)
            x_chunk = np.array_split(X_batch,size,axis = 1)
        else : 
            x_chunk = None
            y_chunk = None 
            model = None 
        ##### Communication #######
        X = comm.scatter(x_chunk, root = 0)
        Y = comm.scatter(y_chunk, root = 0)
        model = comm.bcast(model, root = 0)
        ###########################
        loss = model.step(X,Y)
        for i in range(len(model.layers)): 
            global_sum_w = np.zeros(model.layers[i].dweights_.shape, dtype=float)
            global_sum_b = np.zeros(model.layers[i].dbias_ .shape, dtype=float)
            comm.Reduce(model.layers[i].dweights_, global_sum_w, op=MPI.SUM, root=0)
            comm.Reduce(model.layers[i].dbias_, global_sum_b, op=MPI.SUM, root=0)
            if worker == 0 :
                model.layers[i].dweights_ = global_sum_w/size
                model.layers[i].dbias_ = global_sum_b/size
        global_loss = comm.reduce(loss, op=MPI.SUM, root=0)
        if worker == 0 :
            model.update(optimizer)
            optimizer.ite = optimizer.ite + 1   
            print('loss :', global_loss/size)



