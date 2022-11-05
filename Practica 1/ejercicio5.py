#Estudie el efecto del muestreo aleatorio del 10% de las instancias sin reemplazamiento sobre el árbol de
#decisión y el vecino más cercano. Comente lo resultados. Compare los resultados con un muestreo del mismo
#porcentaje pero estratificado

import pandas as pd
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from scipy.io import arff

#Load dataset
df = arff.loadarff('iris.arff')
df = pd.DataFrame(df[0])

#Convert the class column to numeric
df['class'] = pd.Categorical(df['class'])
df['class'] = df['class'].cat.codes

#Split the dataset into training and testing
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

#Decision Tree
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print ("Error Decision Tree:",1-metrics.accuracy_score(y_test, y_pred))

#Plot the tree
tree.plot_tree(clf)
plt.show()

#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Error KNN:",1-metrics.accuracy_score(y_test, y_pred))