import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('res/iris.csv')
iris = df.iloc[:, 1:]
iris.pop('Species')

sns.heatmap(np.random.rand(10, 10))
sns.clustermap(iris)

plt.show()