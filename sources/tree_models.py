import numpy as np 

class TreeNode : 
    def __init__(self,left_node = None, right_node = None, value = None, 
                 spliting_feature = None, spliting_threshold = None, 
                 depth = None):
        self.value = value 
        self.left_node = left_node
        self.right_node = right_node 
        self.spliting_feature = spliting_feature
        self.spliting_threshold = spliting_threshold
        self.depth = depth
    
    def _split(self, X): 
        '''
        arguments 
        X ::: array(n_samples, n_features) ::: 
        return 
        X_left ::: array (n_left_samples, n_features) :::
        X_right ::: array (n_right_samples, n_features) ::: 
        '''
        boolean_array = X[:,self.spliting_feature] > self.spliting_threshold
        # below than threshold 
        idx_left = np.argwhere(np.logical_not(boolean_array)).flatten()
        # Higher than threshold
        idx_right = np.argwhere(boolean_array).flatten()
        X_left = None 
        X_right = None 
        if len(idx_left) > 0 : 
            X_left = X[idx_left,:]
        if len(idx_right) > 0 :
            X_right = X[idx_right,:]
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
    
    def predict(self,X):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: 
        '''
        n_samples, n_features = X.shape
        result = []
        for i in range(n_samples): 
            x_sample = np.expand_dims(X[i,:],axis = 0)
            print(x_sample)
            val_tmp = self.root._data_flow(x_sample)
            result.append(val_tmp)
        return np.asarray(result)
        