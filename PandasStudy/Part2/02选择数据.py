import numpy as np
import pandas as pd

df = pd.read_csv(r'PandasStudy/res/data01.csv')
print(df)

# select by label
print(df.loc[:3, 'A'])

# select by position
print(df.iloc[1, 3])
print(df.iloc[1, :])
print(df.iloc[:, 3])

# Boolean indexing
print(df[df['A'] > 3])