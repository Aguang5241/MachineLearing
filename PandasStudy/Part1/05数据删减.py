import pandas as pd

data = pd.read_csv('res/los_census.csv')
######################1.DataFrame.drop()#########################
print(data.drop(labels=['Median Age', 'Total Males'], axis=1))
######################2.DataFrame.drop_duplicates()#########################
print(data.drop_duplicates())
######################3.DataFrame.dropna()#########################
print(data.dropna())