import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('res/iris.csv')
iris = df.iloc[:, 1:]
sns.relplot(x="Sepal.Length", y="Sepal.Width", data=iris)
sns.relplot(x="Sepal.Length", y="Sepal.Width", hue='Species', data=iris)
sns.relplot(x="Sepal.Length", y="Sepal.Width", hue='Species', style='Species', data=iris)
# 等同于 sns.scatterplot(x="Sepal.Length", y="Sepal.Width", hue='Species', data=iris)
sns.relplot(x="Sepal.Length", y="Sepal.Width", hue='Species', style='Species',kind='line', data=iris)
# 等同于 sns.lineplot(x="Sepal.Length", y="Sepal.Width", hue='Species', data=iris)
plt.show()