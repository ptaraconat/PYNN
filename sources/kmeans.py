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
        '''
        self.n_classes = n_class
        self.max_iters = max_iters
        self.plot_steps = plot_steps 

        self.clusters = [ [] for _ in range(self.n_classes)]
        # mean of features for each cluster 
        self.centroids = []
    
    def predict(self,x):
        '''
        '''
        self.x = x 
        self.n_samples, self.n_features = x.shape
        # 
        random_indices = np.random.choice(self.n_samples,self.n_classes,replace = False)
        self.centroids = [self.x[index] for index in random_indices]
        #
        for _ in range(self.max_iters):
            # update cluster 
            self.clusters = self._create_cluster(self.centroids)
            #
            if self.plot_steps : 
                self.plot()
            #
            centroid_olds = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            #
            if self._is_converged(centroid_olds,self.centroids) : 
                break
        
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self,clusters):
        '''
        '''
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster : 
                labels[sample_idx] = cluster_idx
        return labels


    def _create_cluster(self,centroids):
        '''
        '''
        clusters = [ [] for _ in range(self.n_classes)]
        for idx, sample in enumerate(self.x):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self,sample, centroids):
        '''
        '''
        distances = [euclidian_distance(sample,point) for point in centroids]
        return np.argmin(distances)

    def _get_centroids(self,clusters):
        '''
        '''
        centroids = np.zeros((self.n_classes,self.n_features))
        for cluster_indx, cluster in enumerate(clusters) :
            mean_tmp = np.mean(self.x[cluster],axis = 0)
            centroids[cluster_indx,:] = mean_tmp 
        return centroids

    def _is_converged(self,centroids_old, centroids) : 
        '''
        '''
        distances = [ euclidian_distance(centroids_old[i],centroids[i]) for i in range(self.n_classes)]
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


if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = X, y = make_blobs(n_samples=500, centers=4, n_features=2,random_state=42)
    print(X.shape)
    model = KMeans(n_class = 4)
    y_pred = model.predict(X)


        
