import sys as sys 
sys.path.append('sources/')
from tree_models import * 
import pytest

@pytest.fixture
def node_regression_fixture():
    node = RegressionNode(left_node = None, right_node = None, value = None, 
                    spliting_feature = 1, 
                    spliting_threshold = 2)
    return node

def test_leaf_value(node_regression_fixture):
    y = np.array([1.1, 2.9, 2])
    leaf_val = node_regression_fixture._get_leaf_value(y)
    assertion = leaf_val == 2.
    assert assertion 
    
def test_information_gain(node_regression_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3],
                     [2.1, 3.2, 1],
                     [6, 3, 2],
                     [4.4, 4, 5]])
    y = np.array([2,4,9,2.,0,10,9,16])
    ig = node_regression_fixture._get_information_gain(y,data[:,1],2)
    score_parent = np.var(y)
    std_child1 = np.var([2,4 , 2, 0])
    std_child2 = np.var([9,10,9,16])
    child_score = 0.5*(std_child1 + std_child2) # expected : 5.25
    assertion = ig == score_parent - child_score
    assert assertion