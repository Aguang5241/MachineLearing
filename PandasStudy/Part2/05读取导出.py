import pandas as pd

df = pd.read_csv('PandasStudy/res/data01.csv')
print(df)

df.to_csv('PandasStudy/res/toData01.csv')