#Discretización K-bins y Feature binarization

import numpy as np
import pandas as pd

from sklearn import tree, datasets
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from scipy.io import arff
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer

# Load dataset
dataset = datasets.load_breast_cancer()
X_dataset = dataset.data

#Binarization
binarizer = Binarizer(threshold=0.0).fit(X_dataset)
binaryX = binarizer.transform(X_dataset)

#Discretization
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit(X_dataset)
X_binned = est.transform(X_dataset)


# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)

#Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print ("Error Decision Tree:",1-metrics.accuracy_score(y_test, y_pred))

#Plot the tree
tree.plot_tree(clf)
#plt.show()

#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Error KNN:",1-metrics.accuracy_score(y_test, y_pred))
