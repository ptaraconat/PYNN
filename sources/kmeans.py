import numpy as np 
import matplotlib.pyplot as plt 

# from P. Loeber 

def euclidian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2.))

class KMeans :
    '''
    '''
    def __init__(self,n_class = 5, max_iters = 100, plot_steps = False):
        '''
        arguments 
        n_class ::: int ::: number of requested clusters 
        max_iters ::: int ::: maximum number of training iterations
        plot_steps ::: bool :::
        updates 
        self.n_classes
        self.max_iters
        self.plot_steps
        self.clusters
        self.centroids
        '''
        self.n_classes = n_class
        self.max_iters = max_iters
        self.plot_steps = plot_steps 

        self.clusters = [ [] for _ in range(self.n_classes)]
        # mean of features for each cluster 
        self.centroids = []

    def _init_centroids(self, X):
        '''
        arguments 
        X ::: array(n_samples, n_features) ::: input_data 
        updates 
        self.centroids ::: 
        '''
        n_samples = np.size(X,0)
        random_indices = np.random.choice(n_samples,
                                          self.n_classes,
                                          replace = False,
                                          ) 
        self.centroids = [X[index] for index in random_indices]
        
    def predict2(self,X):
        '''
        arguments 
        X ::: array(n_samples, n_features) ::: input_data 
        returns : 
        labels ::: arrays(n_samples) ::: predicted classification 
        
        '''
        if self.centroids == []:
            raise Exception("Error : centroids are not initialized")
        else : 
            closest_centroids = self._find_closest_centroids(X)
        return np.squeeze(closest_centroids)
            
            
    def _find_closest_centroids(self,X):
        '''
        arguments 
        X ::: array(n_samples, n_features) of float ::: input_data 
        returns : 
        closest_centroids ::: array (n_samples,1) int ::: indexes of 
        the closest centroids
        '''
        n_samples = np.size(X,0)
        closest_centroids = np.zeros((n_samples,1)).astype(int)
        for i in range(n_samples):
            sample = X[i,:]
            distances = [euclidian_distance(sample,point) for point in self.centroids]
            closest_centroids[i,0]= np.argmin(distances)
        return closest_centroids
    
    def _set_centroids(self,clusters):
        '''
        arguments 
        clusters ::: list (n_classes) of arrays (cluster_size, n_features) :::
        updates 
        self.centroids ::: list (n_classes) of arrays (1,n_features) :::
        '''
        centroids = []
        for cluster in clusters : 
            mean_tmp = np.mean(cluster, axis = 0)
            centroids.append(mean_tmp)
        self.centroids = centroids

    def _is_converged(self,centroids_old, centroids) : 
        '''
        '''
        distances = [ euclidian_distance(centroids_old[i],centroids[i]) 
                     for i in range(self.n_classes)]
        return sum(distances) == 0
    
    def plot(self):
        '''
        ''' 
        fig, ax = plot.subplots(figsize = (12,8))
        for i, index in enumerate(self.clusters):
            point = self.x[index]
            ax.scatter(*point)

        for point in self.centroids : 
            ax.scatter(*point, marker = 'x', markersize = 8, color = 'black')

        plt.show()
            
    def _get_clusters(self,X):
        '''
        arguments 
        X ::: array(n_samples, n_features) of float ::: input_data 
        returns 
        clusters ::: list (n_classes) of arrays (cluster_size,n_features) ::: 
        '''
        labels = self.predict2(X)
        clusters = []
        # classes/clusters loop 
        for i in range(self.n_classes):
            idx_tmp = np.argwhere(labels == i)[:,0]
            clusters.append(X[idx_tmp,:])
        return clusters 
    
    def fit(self,X):
        '''
        arguments 
        X ::: array(n_samples, n_features) of float ::: input_data 
        updates 
        self.centroids ::: list (n_classe) of arrays (n_features)
        '''
        # init centroids 
        self._init_centroids(X)
        # training loop 
        for iter in range(self.max_iters):
            print('iter ',str(iter))
            # split X in clusters
            clusters = self._get_clusters(X)
            # store former centroids 
            centroid_olds = self.centroids
            # update centroids 
            self._set_centroids(clusters)
            # check convergence 
            if self._is_converged(centroid_olds,self.centroids) : 
                break

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=500, centers=4, n_features=2,random_state=42)
    print(X.shape)
    model = KMeans(n_class = 4)
    model.fit(X)
    y_hat = model.predict2(X)
    print(y)
    print(y_hat)


        
