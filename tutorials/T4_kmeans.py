import sys as sys 
sys.path.append('../')
from sources.kmeans import * 
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, centers=4, n_features=2,random_state=42)
print(X.shape)
model = KMeans(n_class = 4)
model.fit(X)
y_hat = model.predict2(X)
print(y)
print(y_hat)