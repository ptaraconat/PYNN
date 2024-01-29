import numpy as np 
import math 
from collections import Counter

# Create entropy function 
def entropy(y):
    hist = np.bincount(y)
    ps = hist/len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class TreeNode : 
    def __init__(self,left_node = None, right_node = None, value = None, 
                 spliting_feature = None, spliting_threshold = None, 
                 depth = 0):
        self.value = value 
        self.left_node = left_node
        self.right_node = right_node 
        self.spliting_feature = spliting_feature
        self.spliting_threshold = spliting_threshold
        self.depth = depth
        
    def _get_split_indexes(self,x_column,threshold):
        '''
        '''
        boolean_array = x_column > threshold
        # below than threshold 
        idx_left = np.argwhere(np.logical_not(boolean_array)).flatten()
        # Higher than threshold
        idx_right = np.argwhere(boolean_array).flatten()
        return idx_left, idx_right
    
    def _split_labels(self,y,X): 
        
        idx_left, idx_right = self._get_split_indexes(X[:,self.spliting_feature], 
                                                      self.spliting_threshold)
        y_left = None 
        y_right = None 
        if len(idx_left) > 0 : 
            y_left = y[idx_left]
        if len(idx_right) > 0 :
            y_right = y[idx_right]
        return y_left, y_right
        
        
    
    def _split(self, X): 
        '''
        arguments 
        X ::: array(n_samples, n_features) ::: 
        return 
        X_left ::: array (n_left_samples, n_features) :::
        X_right ::: array (n_right_samples, n_features) ::: 
        '''
        idx_left, idx_right = self._get_split_indexes(X[:,self.spliting_feature], 
                                                      self.spliting_threshold)
        X_left = None 
        X_right = None 
        if len(idx_left) > 0 : 
            X_left = X[idx_left]
        if len(idx_right) > 0 :
            X_right = X[idx_right]
        return X_left, X_right
    
    def _is_leaf(self):
        '''
        arguments 
        None 
        returns 
        bool ::: bool
        '''
        if self.value != None :
            return True 
        else :
            return False 
    
    def _data_flow(self,X):
        '''
        arguments 
        X ::: array(1, n_features) ::: 
        returns 
        value
        '''
        if self._is_leaf() : 
            return self.value 
        else : 
            X_left, X_right = self._split(X)
            if X_left is not None : 
                return self.left_node._data_flow(X_left)
            if X_right is not None : 
                return self.right_node._data_flow(X_right)
    
    def _get_child_entropy(self,y,x_col,threshold):
        '''
        arguments :
        y ::: array(n_samples) ::: labels table 
        x_col ::: array(n_samples) ::: feature column
        threshold ::: float ::: threshold used to split data given x_col
        returns 
        child_entropy ::: float ::: weighted entropy of child when the node split 
        is achieved on feature x_col with the given threshold. 
        '''
        idx_left, idx_right = self._get_split_indexes(x_col,threshold)
        # weighted average child Entropy 
        n_l, n_r = len(idx_left), len(idx_right)
        n = len(y)
        e_l, e_r = entropy(y[idx_left]), entropy(y[idx_right])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return child_entropy
    
    def _get_information_gain(self, y, x_col, threshold):
        '''
        arguments :
        y ::: array(n_samples) ::: labels table 
        x_col ::: array(n_samples) ::: feature column
        threshold ::: float ::: threshold used to split data given x_col
        returns 
        information_gain ::: float ::: 
        '''
        # parent entropy
        parent_entropy = entropy(y)
        child_entropy = self._get_child_entropy(y, x_col, threshold)
        # information gain 
        ig = parent_entropy - child_entropy
        return ig  
    
    def _get_best_criterion(self,X,y, randomized_features = None):
        '''
        arguments : 
        X ::: array(n_samples,n_features) ::: model input data 
        y ::: array(n_samples) ::: labels array
        returns : 
        split_index ::: int ::: 
        split_threshold ::: float :::
        '''
        best_ig = - 1
        split_index = None 
        split_threshold = None 
        if randomized_features == None :
            indices = np.arange(np.size(X,1))
        else : 
            if randomized_features <= np.size(X,1) : 
                indices = np.random.choice(np.size(X,1),randomized_features, replace = False)
            else : 
                raise Exception("randomized_features is higher than the number of input features")
        print(indices)
        for ind in indices : 
            x_col = X[:,ind]
            thresholds = np.unique(x_col)
            for threshold in thresholds :
                ig = self._get_information_gain(y, x_col, threshold)
                if ig > best_ig : 
                    best_ig = ig 
                    split_index = ind 
                    split_threshold = threshold 
        return split_index, split_threshold 
    
    def _stop_growing(self, y, max_depth, min_samples_required) : 
        n_labels = len(np.unique(y))
        n_samples = len(y) 
        return (self.depth >= max_depth or n_labels == 1 or n_samples < min_samples_required)
    
    def _get_leaf_value(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _set_and_grow(self, X, y, 
                      max_depth = 100, 
                      min_samples_required = 2,
                      randomized_features = None) : 
        
        if self._stop_growing(y, max_depth, min_samples_required) : 
            self.value = self._get_leaf_value(y)
        else : 
            split_index, split_threshold = self._get_best_criterion(X, y, 
                                                                    randomized_features= randomized_features)
            self.spliting_feature = split_index
            self.spliting_threshold = split_threshold
            # split data 
            X_left, X_right = self._split(X)
            y_left, y_right = self._split_labels(y,X) 
            # 
            self.left_node = TreeNode(depth = self.depth + 1)
            self.right_node = TreeNode(depth = self.depth +1)
            #
            self.left_node._set_and_grow(X_left,y_left,
                                         max_depth= max_depth,
                                         min_samples_required= min_samples_required,
                                         randomized_features = randomized_features)
            self.right_node._set_and_grow(X_right,y_right,
                                          max_depth = max_depth,
                                          min_samples_required = min_samples_required,
                                          randomized_features = randomized_features)
            
          
class TreeClassifier : 
    def __init__(self, min_sample_split = 2,max_depth = 100, 
                 n_features = None):
        '''
        arguments 
        min_samples_split ::: int :::
        max_depth ::: int ::: 
        n_features ::: int ::: 
        '''
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None 
        
    def fit(self,X,y):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: 
        y ::: array (nsamples) ::: 
        '''
        pass 
        
    def predict(self,X):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: 
        '''
        n_samples, n_features = X.shape
        result = []
        for i in range(n_samples): 
            x_sample = np.expand_dims(X[i,:],axis = 0)
            val_tmp = self.root._data_flow(x_sample)
            result.append(val_tmp)
        return np.asarray(result)
        