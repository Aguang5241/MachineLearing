import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('res/iris.csv')
iris = df.iloc[:, 1:]
sns.distplot(iris['Sepal.Length'])
sns.jointplot(x='Sepal.Length', y='Sepal.Width', data=iris)
sns.jointplot(x='Sepal.Length', y='Sepal.Width', data=iris, kind='kde')
sns.jointplot(x='Sepal.Length', y='Sepal.Width', data=iris, kind='hex')
sns.jointplot(x='Sepal.Length', y='Sepal.Width', data=iris, kind='reg')
sns.pairplot(iris)
sns.pairplot(iris, hue='Species')

plt.show()