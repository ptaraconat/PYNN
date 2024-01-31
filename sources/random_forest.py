import numpy as np 
from sources.tree_models import TreeClassifier
from collections import Counter
# Resources : P. Loeber
def bootstrap_sample(X,y):
	n_samples = X.shape[0]
	idxs = np.random.choice(n_samples,size = n_samples, replace = True)
	return X[idxs,:], y[idxs]

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
            tree = TreeClassifier(min_sample_split = self.min_samples_splits, 
                                  max_depth = self.max_depth,  
                                  randomized_features = self.n_feats)
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
            counter = Counter(tree_preds[i,:])
            most_common = counter.most_common(1)[0][0]
            labels.append(most_common)
        return np.asarray(labels)