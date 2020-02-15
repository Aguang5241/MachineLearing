import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('res/iris.csv')
iris = df.iloc[:, 1:]
sns.regplot(x='Sepal.Length', y='Sepal.Width', data=iris)
sns.lmplot(x='Sepal.Length', y='Sepal.Width', data=iris, hue='Species')

plt.show()