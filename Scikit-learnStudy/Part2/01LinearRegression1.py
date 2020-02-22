from sklearn import linear_model

reg = linear_model.LinearRegression()

x = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
print(reg.fit(x, y))
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

print(reg.coef_)
# [0.5 0.5]