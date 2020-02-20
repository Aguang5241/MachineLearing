import numpy as np
import pandas as pd

df1 = pd.DataFrame(np.ones((2, 3)) * 1, columns=['a', 'b', 'c'])
df2 = pd.DataFrame(np.ones((2, 3)) * 2, columns=['a', 'b', 'c'])
df3 = pd.DataFrame(np.ones((2, 3)) * 3, columns=['a', 'b', 'c'])

print(df1)
print(df2)
print(df3)

res = pd.concat([df1, df2, df3], axis=0)
print(res)
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res)

df01 = pd.DataFrame(np.ones((3, 3)) * 0, columns=['a', 'b', 'c'], index=[1, 2, 3])
df02 = pd.DataFrame(np.ones((3, 3)) * 1, columns=['b', 'c', 'd'], index=[2, 3, 4])
resOuter = pd.concat([df01, df02], join='outer', ignore_index=True)
resInner = pd.concat([df01, df02], join='inner', ignore_index=True)
print(resOuter)
print(resInner)