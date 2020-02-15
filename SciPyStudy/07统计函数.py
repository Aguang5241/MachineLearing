import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import describe, norm

sns.set()
plt.hist(norm.rvs(size=1000))

x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 1000)
plt.plot(x, norm.pdf(x))

print(describe(x))
# DescribeResult(nobs=1000, minmax=(-2.3263478740408408, 2.3263478740408408), mean=0.0, variance=1.8093857372505617, skewness=9.356107558947013e-17, kurtosis=-1.2000024000024)

plt.show()
