import sys as sys 
sys.path.append('sources/')
from tree_models import * 
import pytest

@pytest.fixture
def node_fixture():
    node = TreeNode(left_node = None, right_node = None, value = None, 
                    spliting_feature = 1, 
                    spliting_threshold = 2)
    return node

@pytest.fixture
def node_classif_fixture():
    node = ClassifNode(left_node = None, right_node = None, value = None, 
                    spliting_feature = 1, 
                    spliting_threshold = 2)
    return node

@pytest.fixture 
def node_fixture2():
    node_left = TreeNode(left_node = None, right_node = None, value = 5, 
                         spliting_feature = None, 
                         spliting_threshold = None,
                         depth = 1)
    node_right =  TreeNode(left_node = None, right_node = None, value = 11, 
                           spliting_feature = None, 
                           spliting_threshold = None,
                           depth = 1)
    node = TreeNode(left_node = node_left, right_node = node_right, value = None, 
                    spliting_feature = 1, 
                    spliting_threshold = 2)
    return node 
 
@pytest.fixture
def tree_fixture():
    node_left = TreeNode(left_node = None, right_node = None, value = 5, 
                         spliting_feature = None, 
                         spliting_threshold = None,
                         depth = 1)
    node_right =  TreeNode(left_node = None, right_node = None, value = 11, 
                           spliting_feature = None, 
                           spliting_threshold = None,
                           depth = 1)
    node = TreeNode(left_node = node_left, right_node = node_right, value = None, 
                    spliting_feature = 1, 
                    spliting_threshold = 2)
    tree = TreeClassifier(min_sample_split = 2,max_depth = 100, n_features = None)
    tree.root = node
    return tree
       

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
    
def test_node_flow(node_fixture2):
    data = np.array([[1.1, 1, 4]])
    value = node_fixture2._data_flow(data)
    exp_value = 5
    assertion = value == exp_value
    assert assertion 

def test_node_flow2(tree_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.4, 1, 2],
                     [1., 2.4, 4]])
    predictions = tree_fixture.predict(data)
    print(predictions)
    expected_outcome = np.array([5,5,11])
    assertion = np.all(predictions == expected_outcome)
    assert assertion
    
def test_entropy():
    data = np.array([1,1,1,1,1,0,0,0,0,0])
    ent = entropy(data)
    assertion = ent == 1
    assert assertion 
    
def test_child_entropy(node_classif_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3],
                     [2.1, 3.2, 1],
                     [6, 3, 2],
                     [4.4, 4, 5],
                     [4.4, 4, 5]])
    y = np.array([0,0,1,0,0,1,1,1,1])
    child_entropy = node_classif_fixture._get_child_score(y,data[:,1],2)
    print(child_entropy)
    assertion = child_entropy == 0
    assert assertion 
    
def test_information_gain(node_classif_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3],
                     [2.1, 3.2, 1],
                     [6, 3, 2],
                     [4.4, 4, 5]])
    y = np.array([0,0,1,0,0,1,1,1])
    ig = node_classif_fixture._get_information_gain(y,data[:,1],2)
    assertion = ig == 1
    assert assertion
    
def test_best_crit(node_classif_fixture): 
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3],
                     [2.1, 3.2, 1],
                     [6, 3, 2],
                     [4.4, 4, 5]])
    y = np.array([0,0,1,0,0,1,1,1])
    split_index, split_threshold = node_classif_fixture._get_best_criterion(data,y, randomized_features = 3)
    assertion = split_index == 1 and split_threshold == 2.0 
    assert assertion 

def test_get_value(node_classif_fixture): 
    y = np.array([0,0,0,1,0,0,1,1,1,1,1])
    leaf_label = node_classif_fixture._get_leaf_value(y)
    print(leaf_label)
    assertion = leaf_label == 1 
    assert assertion 

def test_node_grow(node_classif_fixture):
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3],
                     [2.1, 3.2, 1],
                     [6, 3, 2],
                     [4.4, 4, 5]])
    y = np.array([0,0,1,0,0,1,1,1])
    node_classif_fixture._set_and_grow(data,y,
                               max_depth = 3,
                               min_samples_required = 2,
                               randomized_features = 3)
    
    val_left = node_classif_fixture.left_node.value
    val_right = node_classif_fixture.right_node.value
    assertion = val_left == 0 and val_right == 1
    assert assertion
    
def test_tree_fit():
    data = np.array([[1.1, 1, 4],
                     [1.2, 2, 3],
                     [2.3, 3, 4],
                     [2.9, 1.1, 5],
                     [3.5, 0.1, 3],
                     [2.1, 3.2, 1],
                     [6, 3, 2],
                     [4.4, 4, 5]])
    y = np.array([0,0,1,0,0,1,1,1])
    tree = TreeClassifier(min_sample_split = 2,max_depth = 3, 
                          n_features = None, randomized_features = 3)
    tree.fit(data,y)
    val_left = tree.root.left_node.value
    val_right = tree.root.right_node.value
    assertion = val_left == 0 and val_right == 1
    assert assertion
    