import sys as sys 
sys.path.append('../sources/')
from kmeans import * 
import pytest

@pytest.fixture
def kmeans_model():
    model = KMeans(n_class = 3)
    return model
@pytest.fixture
def kmeans_model2():
    model = KMeans(n_class = 3)
    model.centroids = [np.array([1, 1]), 
                       np.array([3, 3]), 
                       np.array([2, 2])]
    return model

def test_init_centroids(kmeans_model):
    input = np.array([[1,1],[2,2],[3,3]])
    kmeans_model._init_centroids(input)
    assertion = np.all(np.unique(kmeans_model.centroids,axis = 0) == input)
    assert assertion
    
def test_find_closest(kmeans_model2):
    input = np.array([[1,1],[2,2],[3,3],[3.1,3.1]])
    closest_centroids = kmeans_model2._find_closest_centroids(input)
    expected_val = np.array([[0],[2],[1],[1]]).astype(int)
    assertion = np.all(closest_centroids == expected_val)
    assert assertion

def test_prediction(kmeans_model2): 
    input = np.array([[1,1],[2,2],[3,3],[3.1,3.1]])
    closest_centroids = kmeans_model2.predict2(input)
    expected_val = np.array([0,2,1,1]).astype(int)
    assertion = np.all(closest_centroids == expected_val)
    assert assertion 
    
    