import sys as sys 
sys.path.append('../')
from sources.mpi import * 

comm = MPI.COMM_WORLD
worker = comm.Get_rank()

if worker == 0 :
    # Data 
    Ndata = 1000
    X_data = np.linspace(0,3,Ndata) 
    X_data = np.expand_dims(X_data,0)
    Y_data = np.power(X_data,2)
    # model
    layer1 = Dense(4,activation ='sigmoid',input_units = 1)
    layer2 = Dense(1,activation ='linear',input_units = 4)
    model = Model(layers = [layer1,layer2],loss = 'MSE')
    # optimizer
    optimizer = Adam(learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8)
else :
    X_data = None 
    Y_data = None
    optimizer = None
    model = None
model = mpi_model_trainer(X_data,Y_data,model,optimizer,n_epochs=10000,batch_size = 200)
if worker == 0 :
    print(Y_data[:,0:10])
    print(model.predict(X_data[:,0:10]))

