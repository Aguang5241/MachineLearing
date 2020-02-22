from sklearn import linear_model

reg = linear_model.Ridge(alpha=0.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg.coef_)
# [0.34545455 0.34545455]
print(reg.intercept_)
# 0.13636363636363638