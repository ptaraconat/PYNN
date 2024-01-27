import sys as sys 
sys.path.append('../sources/')
from tree_models import * 
import pytest

@pytest.fixture
def node_fixture():
    node = TreeNode(left_node = None, right_node = None, value = None, 
                    spliting_feature = 1, 
                    spliting_threshold = 2)
    return node 

def test_node_split(node_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3]])
    exp_left = np.array([[1.1, 1, 4],
                         [1.2, 2, 3],
                         [2.9, 1.1, 5],
                         [3.5, 0.1, 3]])
    exp_right = np.array([[2.3, 3, 4]])
    data_left, data_right = node_fixture._split(data)
    assertion = True 
    assertion = assertion and np.all(data_left == exp_left)
    assertion = assertion and np.all(data_right == exp_right)
    assert assertion
    
def test_node_split2(node_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 0, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3]])
    exp_left = data 
    exp_right = None
    data_left, data_right = node_fixture._split(data)
    print(data_left)
    print(data_right)
    assertion = True
    assertion = assertion and np.all(data_left == exp_left)
    assertion = assertion and np.all(data_right == exp_right)
    assert assertion