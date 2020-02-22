import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

# 希尔伯特矩阵（Hilbert matrix），是一种数学变换矩阵，正定且高度变态：
# 任何一个元素发生一点变化，整个矩阵的行列式的值和逆矩阵都会发生巨大的变化
X = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
Y = np.ones(10)
# 指定 alpha 个数
nAlphas = 200
# 指定 alpha
alphas = np.logspace(-10, -2, nAlphas)
# 使用不同的 alpha 去训练，并获得对于的参数列表
coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, Y)
    coefs.append(ridge.coef_)
# 使用交叉验证方法自动最优化
clf = linear_model.RidgeCV(fit_intercept=False)
clf.fit(X, Y)
# 获得最有 alpha 和对应参数
print('The best alpha:\n', clf.alpha_)
print('The coefficients:\n', clf.coef_)
# 绘图
sns.set()
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])

plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')

plt.show()
