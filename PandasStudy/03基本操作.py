import pandas as pd
# import numpy as np

data = pd.read_csv('res/los_census.csv')
#########################1.默认显示前/后5行##########################
print(data.head())
print(data.tail())
#########################2.对数据集进行概览##########################
print(data.describe())
#########################3.将DataFrame转换为Numpy数组##########################
print(data.values)
#########################4.其他属性##########################
print(data.index)
# RangeIndex(start=0, stop=319, step=1)
print(data.columns)
# Index(['Zip Code', 'Total Population', 'Median Age', 'Total Males',
#        'Total Females', 'Total Households', 'Average Household Size'],
#       dtype='object')
print(data.shape)
# (319, 7)