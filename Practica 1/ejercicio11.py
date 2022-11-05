# Correlation matrix

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

# Load the dataset
dataset = datasets.load_iris()

# Correlation matrix
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation matrix Iris dataset')
plt.show()