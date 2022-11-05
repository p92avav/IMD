# Matricial Scatter Plot of the datasets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

# Load the dataset
dataset = datasets.load_iris()

# Scatter plot
sns.pairplot(pd.DataFrame(dataset.data, columns=dataset.feature_names))
plt.show()