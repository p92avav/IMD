# Ejercicio 2 pero normalizando

from sklearn import tree, datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Load dataset
dataset = datasets.load_iris()

# Normalize the dataset interval [0,1]
#dataset.data = preprocessing.normalize(dataset.data)

# Standardize the dataset with mean = 0 and std = 1
#dataset.data = preprocessing.scale(dataset.data)


# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)

# Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Error Decision Tree:", 1 - metrics.accuracy_score(y_test, y_pred))

# Plot the tree
tree.plot_tree(clf)
# plt.show()

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Error KNN:", 1 - metrics.accuracy_score(y_test, y_pred))
