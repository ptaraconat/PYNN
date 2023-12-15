import sys 
sys.path.append('../sources/')
from NN_numerics import * 
import pytest

@pytest.fixture
def layer(n_unists,input_units):
    layer = Dense(n_units,input_units = input_units)
    return layer
@pytest.fixture
def layer1():
    layer = Dense(2,input_units = 3,activation ='linear')
    W = np.array([[1,2,3],[2,4,6]])
    B = np.array([[4],[0]])
    layer.bias = B 
    layer.weights = W
    return layer
@pytest.fixture
def layer2():
    layer = Dense(3,input_units = 2,activation ='linear')
    W = np.array([[1,2],[2,4],[3,6]])
    B = np.array([[4],[0],[1]])
    layer.bias = B 
    layer.weights = W
    return layer 


def test_layer_forward(layer1):
    input = np.array([[1],[1],[1]])
    layer1.forward(input)
    bool_tmp = np.all(layer1.cache['A'] == np.array([[10],[12]]))
    assert bool_tmp
    
def test_model_forward(layer1,layer2):
    model = Model([layer1,layer2],loss = 'MSE')
    input = np.array([[1],[1],[1]])
    layer1.forward(input)
    lay1_out = layer1.cache['A']
    layer2.forward(lay1_out)
    lay2_out = layer2.cache['A']
    model_out = model.predict(input)
    bool_tmp = np.all(model_out == lay2_out)
    assert bool_tmp

