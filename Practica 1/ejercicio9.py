# Box plots de los datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Load dataset
dataset = datasets.load_iris()

# Box plot
plt.figure(1)
plt.boxplot(dataset.data)
plt.title('BoxPlot Iris Dataset')

plt.xticks([1, 2, 3, 4], ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.ylabel('cm')
plt.show()

# Box plot de las clases de Iris
plt.figure(2)
plt.boxplot(dataset.data[dataset.target==0])
plt.title('BoxPlot Iris Dataset - Setosa')
plt.xticks([1, 2, 3, 4], ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.ylabel('cm')
plt.show()

plt.figure(3)
plt.boxplot(dataset.data[dataset.target==1])
plt.title('BoxPlot Iris Dataset - Versicolor')
plt.xticks([1, 2, 3, 4], ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.ylabel('cm')
plt.show()

plt.figure(4)
plt.boxplot(dataset.data[dataset.target==2])
plt.title('BoxPlot Iris Dataset - Virginica')
plt.xticks([1, 2, 3, 4], ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
plt.ylabel('cm')
plt.show()

