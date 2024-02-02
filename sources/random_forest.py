import numpy as np 
from sources.tree_models import TreeClassifier, TreeRegressor, TreeModel, TreeNode
from collections import Counter
# Resources : P. Loeber
def bootstrap_sample(X,y):
	n_samples = X.shape[0]
	idxs = np.random.choice(n_samples,size = n_samples, replace = True)
	return X[idxs,:], y[idxs]

def bootstrap_sample_wweights(X,y,weights,portion):
    n_samples = X.shape[0]
    n_drawn = np.int(np.round(n_samples*portion))
    idxs = np.random.choice(n_samples, 
                           size = n_drawn, 
                           replace = True, 
                           p = weights)
    return X[idxs,:], y[idxs], weights[idxs]

def most_common(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

def average(y):
    return np.mean(y)

def mean_square_error(y,y_hat):
    return np.mean(squared_error(y, y_hat))

def squared_error(y, y_hat): 
    return (error(y,y_hat) ** 2) / 2

def error(y,y_hat):
    return np.squeeze(y)-np.squeeze(y_hat)

class RandomForest:
    def __init__(self,n_trees = 100,min_samples_splits = 100, 
                 max_depth = 100,n_feats = None):
        '''
        arguments 
        n_tress ::: int ::: number of trees in Random Forest 
        min_samples_splits ::: int ::: min number of samples in tree leafs
        max_depth ::: int ::: maximal depth of trees 
        n_feats ::: int ::: number of random features used for
        finding the best node splitting 
        '''
        self.n_trees = n_trees 
        self.min_samples_splits = min_samples_splits
        self.max_depth = max_depth
        self.n_feats = n_feats 
        self.trees = []
        
    def fit(self, X,y):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: Model input data 
        y ::: array (n_samples) ::: labels array 
        updates ::: 
        self.trees ::: list (n_trees) of decision trees ::: 
        '''
        self.trees = []
        for _ in range(self.n_trees):
            tree = self._init_tree()
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)
            
    def predict(self,X):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: Model input data
        returns 
        labels ::: array (n_samples) ::: predicted labels 
        '''
        tree_preds = np.transpose(np.array([tree.predict(X) for tree in self.trees]))
        labels = []
        for i in range(np.size(tree_preds,0)):
            labels.append(self.gather(tree_preds[i,:]))
        return np.asarray(labels)
    
class GradientBoostedTrees(RandomForest):
    def __init__(self,n_trees = 100, min_samples_splits = 100,
                 max_depth = 100, n_feats = None, learning_rate = 0.1):
        '''
        arguments 
        n_tress ::: int ::: number of trees in Random Forest 
        min_samples_splits ::: int ::: min number of samples in tree leafs
        max_depth ::: int ::: maximal depth of trees 
        n_feats ::: int ::: number of random features used for
        finding the best node splitting 
        '''
        super().__init__(n_trees = n_trees,
                         min_samples_splits = min_samples_splits,
                         max_depth = max_depth,
                         n_feats = n_feats)
        self.learning_rate = learning_rate
        self.loss = error
        
    def _get_initial_tree(self,y):
        '''
        arguments : 
        y ::: array (n_samples) ::: labels array 
        returns 
        tree ::: DecisionTree ::: Tree with a single leaf 
        '''
        # Set the (single) leaf node of the tree 
        mean_y = average(y)
        node = TreeNode(left_node = None, right_node = None, value = mean_y, 
                        spliting_feature = None, spliting_threshold = None, 
                        depth = 0)
        # Set tree 
        tree = TreeModel(min_sample_split = 2,max_depth = 0, 
                         n_features = None, randomized_features = None)
        tree.root = node 
        return tree 
    
    def _init_tree(self): 
        '''
        initialise a decision from the RandomForest arguments
        returns :
        tree ::: TreeNode :::  
        '''
        tree = TreeRegressor(min_sample_split = self.min_samples_splits, 
                             max_depth = self.max_depth,  
                             randomized_features = self.n_feats)
        return tree 
    
    def fit(self,X,y):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: Model input data 
        y ::: array (n_samples) ::: labels array 
        updates ::: 
        self.trees ::: list (n_trees) of decision trees ::: 
        '''
        # first tree 
        current_tree = self._get_initial_tree(y)
        self.trees.append(current_tree)
        # calc loss/residuals 
        forest_prediction = current_tree.predict(X)
        residuals = self.loss(y,forest_prediction)
        # Loop 
        res = []
        for i in range(self.n_trees) : 
            # set new tree 
            current_tree = self._init_tree()
            # train the new tree to learn the residuals 
            current_tree.fit(X,residuals)
            # append to random forest 
            self.trees.append(current_tree)
            # update forest prediction
            current_tree_predictions = current_tree.predict(X) 
            forest_prediction = forest_prediction + self.learning_rate * current_tree_predictions  
            # update residuals 
            residuals = self.loss(y,forest_prediction)
            print('Tree no ',i)
            print('MSE : ',mean_square_error(y,forest_prediction))
            res.append(mean_square_error(y,forest_prediction))
        return res 
            
    
    def predict(self,X):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: Model input data
        returns 
        predictions ::: array (n_samples) ::: predictions
        '''
        for i, tree in enumerate(self.trees) : 
            if i == 0 : 
                predictions = tree.predict(X)
            else :
                predictions += self.learning_rate * tree.predict(X)
        return predictions      

class AdaBoostedTree(RandomForest):
    def __init__(self,n_trees = 100):
        '''
        arguments 
        n_tress ::: int ::: number of trees in Random Forest 
        '''
        super().__init__(n_trees = n_trees,
                         min_samples_splits = 2,
                         max_depth = 1,
                         n_feats = None)
        self.tree_weights = []
    
    def fit(self,X,y):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: Model input data 
        y ::: array (n_samples) ::: labels array 
        updates ::: 
        self.trees ::: list (n_trees) of decision trees ::: 
        '''
        n_samples, n_features = X.shape
        # set n_feats in order randomized features in trees 
        self.n_feats = np.int(np.round(n_features*0.8))
        # init samples weights 
        weights = np.ones(n_samples)*(1/n_samples)
        for i in range(self.n_trees): 
            # samples data 
            X_samp, y_samp, weights_samp = bootstrap_sample_wweights(X,y,weights,0.8)
            # tree model 
            tree = TreeClassifier(min_sample_split= self.min_samples_splits,
                                  max_depth = self.max_depth,
                                  randomized_features = self.n_feats)
            tree.fit(X_samp,y_samp)
            # update samples weights and set model weight
            y_pred = tree.predict(X)
            total_error = np.sum((y_pred != y) * weights)
            # calc tree weight
            tree_weight = 0.5* np.log((1-total_error)/(total_error))
            # calc new samples weights 
            ## incorrectly classified case 
            bool_tmp = y_pred != y
            idxs = np.where(bool_tmp)
            weights[idxs] = weights[idxs]*np.exp(tree_weight)
            ## correctly classified case 
            idxs = np.where(np.logical_not(bool_tmp))
            weights[idxs] = weights[idxs]*np.exp(-1.*tree_weight)
            # Normalize sample weights  
            weights /= np.sum(weights)
            # Update forest 
            self.trees.append(tree)
            self.tree_weights.append(tree_weight)
            #
            print(total_error)
    
    def predict(self,X):
        '''
        arguments 
        X ::: array (n_samples, n_features) ::: Model input data
        returns 
        predictions ::: array (n_samples) ::: predictions
        '''
        n_samples = X.shape[0]
        predictions = np.transpose(np.asarray([weight*(2*tree.predict(X)-1)
                                               for weight,tree in zip(self.tree_weights,self.trees)]))
        predictions = np.sign(np.sum(predictions,axis = 1)) > 0
        
        return predictions #np.where(predictions >= 0.5, 1, 0)
        
               
        
class RandomForestClassifier(RandomForest):
    def __init__(self,n_trees = 100,min_samples_splits = 100, 
                 max_depth = 100,n_feats = None):
        '''
        arguments 
        n_tress ::: int ::: number of trees in Random Forest 
        min_samples_splits ::: int ::: min number of samples in tree leafs
        max_depth ::: int ::: maximal depth of trees 
        n_feats ::: int ::: number of random features used for
        finding the best node splitting 
        '''
        super().__init__(n_trees = n_trees,
                         min_samples_splits = min_samples_splits,
                         max_depth = max_depth,
                         n_feats = n_feats)
        self.gather = most_common
        
    def _init_tree(self): 
        '''
        initialise a decision from the RandomForest arguments
        returns :
        tree ::: TreeNode :::  
        '''
        tree = TreeClassifier(min_sample_split = self.min_samples_splits, 
                              max_depth = self.max_depth,  
                              randomized_features = self.n_feats)
        return tree 

class RandomForestRegressor(RandomForest):
    def __init__(self,n_trees = 100,min_samples_splits = 100, 
                 max_depth = 100,n_feats = None):
        '''
        arguments 
        n_tress ::: int ::: number of trees in Random Forest 
        min_samples_splits ::: int ::: min number of samples in tree leafs
        max_depth ::: int ::: maximal depth of trees 
        n_feats ::: int ::: number of random features used for
        finding the best node splitting 
        '''
        super().__init__(n_trees = n_trees,
                         min_samples_splits = min_samples_splits,
                         max_depth = max_depth,
                         n_feats = n_feats)
        self.gather = average
    
    def _init_tree(self): 
        '''
        initialise a decision from the RandomForest arguments
        returns :
        tree ::: TreeNode :::  
        '''
        tree = TreeRegressor(min_sample_split = self.min_samples_splits, 
                             max_depth = self.max_depth,  
                             randomized_features = self.n_feats)
        return tree 