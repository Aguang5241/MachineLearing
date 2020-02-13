import pandas as pd
import numpy as np

#################生成示例DataFrame数据集######################
df = pd.DataFrame(np.random.rand(9, 5), columns=list('ABCDE'))
df.insert(0, 'Time', pd.Timestamp('2020-2-13'))
df.iloc[[1, 3, 5, 7], [0, 2, 4]] = np.nan
df.iloc[[2, 4, 6, 8], [1, 3, 5]] = np.nan
print(df)
#         Time         A         B         C         D         E
# 0 2020-02-13  0.477160  0.732205  0.850030  0.893766  0.250068
# 1        NaT  0.482808       NaN  0.647220       NaN  0.826963
# 2 2020-02-13       NaN  0.953391       NaN  0.074829       NaN
# 3        NaT  0.379712       NaN  0.798543       NaN  0.259623
# 4 2020-02-13       NaN  0.684412       NaN  0.034089       NaN
# 5        NaT  0.313518       NaN  0.130181       NaN  0.978715
# 6 2020-02-13       NaN  0.760904       NaN  0.643206       NaN
# 7        NaT  0.531707       NaN  0.055104       NaN  0.390785
# 8 2020-02-13       NaN  0.870766       NaN  0.924815       NaN
#################1.检测缺失值######################
print(df.isna())
#     Time      A      B      C      D      E
# 0  False  False  False  False  False  False
# 1   True  False   True  False   True  False
# 2  False   True  False   True  False   True
# 3   True  False   True  False   True  False
# 4  False   True  False   True  False   True
# 5   True  False   True  False   True  False
# 6  False   True  False   True  False   True
# 7   True  False   True  False   True  False
# 8  False   True  False   True  False   True
#################2.去除含有缺失值的行/列######################
print(df.dropna())
#         Time       A         B         C         D         E
# 0 2020-02-13  0.9497  0.514973  0.443469  0.397692  0.637217
#################3.使用0填充缺失值######################
print(df.fillna(0))
#                   Time         A         B         C         D         E
# 0  2020-02-13 00:00:00  0.949700  0.514973  0.443469  0.397692  0.637217
# 1                    0  0.799112  0.000000  0.768144  0.000000  0.541273
# 2  2020-02-13 00:00:00  0.000000  0.994804  0.000000  0.935595  0.000000
# 3                    0  0.274625  0.000000  0.335299  0.000000  0.751197
# 4  2020-02-13 00:00:00  0.000000  0.700554  0.000000  0.152000  0.000000
# 5                    0  0.373676  0.000000  0.705303  0.000000  0.675944
# 6  2020-02-13 00:00:00  0.000000  0.432548  0.000000  0.636303  0.000000
# 7                    0  0.650404  0.000000  0.195951  0.000000  0.000636
# 8  2020-02-13 00:00:00  0.000000  0.048306  0.000000  0.726971  0.000000
#################4.使用缺失值前面的值填充缺失值######################
print(df.fillna(method='pad', limit=1))
#         Time         A         B         C         D         E
# 0 2020-02-13  0.949700  0.514973  0.443469  0.397692  0.637217
# 1 2020-02-13  0.799112  0.514973  0.768144  0.397692  0.541273
# 2 2020-02-13  0.799112  0.994804  0.768144  0.935595  0.541273
# 3 2020-02-13  0.274625  0.994804  0.335299  0.935595  0.751197
# 4 2020-02-13  0.274625  0.700554  0.335299  0.152000  0.751197
# 5 2020-02-13  0.373676  0.700554  0.705303  0.152000  0.675944
# 6 2020-02-13  0.373676  0.432548  0.705303  0.636303  0.675944
# 7 2020-02-13  0.650404  0.432548  0.195951  0.636303  0.000636
# 8 2020-02-13  0.650404  0.048306  0.195951  0.726971  0.000636
#################5.使用缺失值后面的值填充缺失值######################
print(df.fillna(method='bfill', limit=1))
#         Time         A         B         C         D         E
# 0 2020-02-13  0.527895  0.075550  0.011263  0.388663  0.185090
# 1 2020-02-13  0.553448  0.957457  0.679721  0.313866  0.827648
# 2 2020-02-13  0.530548  0.957457  0.357661  0.313866  0.695714
# 3 2020-02-13  0.530548  0.628198  0.357661  0.550058  0.695714
# 4 2020-02-13  0.546305  0.628198  0.802658  0.550058  0.978719
# 5 2020-02-13  0.546305  0.804538  0.802658  0.425110  0.978719
# 6 2020-02-13  0.623238  0.804538  0.951752  0.425110  0.017059
# 7 2020-02-13  0.623238  0.648637  0.951752  0.789823  0.017059
# 8 2020-02-13       NaN  0.648637       NaN  0.789823       NaN
#################6.使用平均值填充######################
print(df.fillna(df.mean()['C':'E']))
#         Time         A         B         C         D         E
# 0 2020-02-13  0.755147  0.805999  0.552410  0.964294  0.767257
# 1        NaT  0.271534       NaN  0.421322  0.462191  0.320706
# 2 2020-02-13       NaN  0.661487  0.405653  0.030253  0.401204
# 3        NaT  0.656952       NaN  0.017693  0.462191  0.228412
# 4 2020-02-13       NaN  0.550506  0.405653  0.451707  0.401204
# 5        NaT  0.868886       NaN  0.594137  0.462191  0.175234
# 6 2020-02-13       NaN  0.162444  0.405653  0.407775  0.401204
# 7        NaT  0.657845       NaN  0.442704  0.462191  0.514409
# 8 2020-02-13       NaN  0.284683  0.405653  0.456924  0.401204
#################7.插值填充######################
#################生成示例DataFrame数据集######################
df2 = pd.DataFrame({'A': [1.1, 2.2, np.nan, 4.5, 5.7, 6.9],
                    'B': [.21, np.nan, np.nan, 3.1, 11.7, 13.2]})
print(df2)
#      A      B
# 0  1.1   0.21
# 1  2.2    NaN
# 2  NaN    NaN
# 3  4.5   3.10
# 4  5.7  11.70
# 5  6.9  13.20
print(df2.interpolate())
#       A          B
# 0  1.10   0.210000
# 1  2.20   1.173333
# 2  3.35   2.136667
# 3  4.50   3.100000
# 4  5.70  11.700000
# 5  6.90  13.200000