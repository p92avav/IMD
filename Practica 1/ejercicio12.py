# Correlation matrix transpose

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Load dataset
iris = datasets.load_iris()

# Plot correlation matrix
sns.heatmap(pd.DataFrame(iris.data, columns=iris.feature_names).corr(), annot=True)
plt.title('Correlation matrix Iris dataset')
plt.show()
