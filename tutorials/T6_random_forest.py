# Resources : P. Loeber
import sys as sys 
sys.path.append('../')
from sources.random_forest import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np 

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RandomForestClassifier(n_trees=20, max_depth=10,n_feats = 10)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(np.shape(X_train))
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)