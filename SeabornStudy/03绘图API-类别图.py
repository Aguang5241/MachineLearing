import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('res/iris.csv')
iris = df.iloc[:, 1:]
sns.catplot(x="Sepal.Length", y="Species", data=iris)
# 等同于 sns.catplot(x="Sepal.Length", y="Species", kind='strip', data=iris)
sns.catplot(x="Sepal.Length", y="Species", kind='swarm', data=iris)
sns.catplot(x="Sepal.Length", y="Species", kind='box', data=iris)
sns.catplot(x="Sepal.Length", y="Species", kind='violin', data=iris)
sns.catplot(x="Sepal.Length", y="Species", kind='boxen', data=iris)
sns.catplot(x="Sepal.Length", y="Species", kind='point', data=iris)
sns.catplot(x="Sepal.Length", y="Species", kind='bar', data=iris)
sns.catplot(x="Species", kind='count', data=iris)

plt.show()