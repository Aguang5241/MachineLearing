import numpy as np
import pandas as pd

df = pd.read_csv('PandasStudy/res/data01.csv')
print(df)

df.iloc[2, 2] = 111
print(df)

df.loc[:, 'F'] = 222
print(df)

df.F = np.nan
print(df)