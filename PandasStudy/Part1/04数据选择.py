import pandas as pd

data = pd.read_csv('res/los_census.csv')
######################一、基于索引选择#########################
######################1.选择前三行数据#########################
print(data.iloc[:3])
######################2.选择特定的一行#########################
print(data.iloc[5])
######################3.选择特定的几行#########################
print(data.iloc[[1, 3, 5]])
######################4.选择连续的几列#########################
print(data.iloc[:, 1:4])
######################二、基于标签名选择#########################
######################5.选择前三行#########################
print(data.loc[0:2])
######################6.选择特定的几行#########################
print(data.loc[[1, 3, 5]])
######################7.选择连续的几列#########################
print(data.loc[:, 'Total Population':'Total Males'])
######################7.选择特定的某列某行#########################
print(data.loc[[0, 2], 'Median Age':])