import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import r2_score

np.random.seed(42)
nSamples, nFeatures = 50, 100
X = np.random.randn(nSamples, nFeatures)

idx = np.arange(nFeatures)
coef = (-1) ** idx *np.exp(-idx / 10)
coef[10:] = 0
Y = np.dot(X, coef)

Y += 0.01 * np.random.normal(size=nSamples)

trainX, trainY = X[:nSamples // 2], Y[:nSamples // 2]
testX, testY = X[nSamples // 2:], Y[nSamples // 2:]

alpha = 0.1
lasso = Lasso(alpha=alpha)
predictYLasso = lasso.fit(trainX, trainY).predict(testX)
r2ScoreLasso = r2_score(testY, predictYLasso)
print('r2_score of Lasso: %.2f' % r2ScoreLasso)

elasticNet = ElasticNet(alpha=alpha, l1_ratio=0.7)
predictYElasticNet = elasticNet.fit(trainX, trainY).predict(testX)
r2ScoreElasticNet = r2_score(testY, predictYElasticNet)
print('r2_score of ElasticNet: %.2f' % r2ScoreElasticNet)

m, s, _ = plt.stem(np.where(elasticNet.coef_)[0], elasticNet.coef_[elasticNet.coef_ != 0],
                   markerfmt='x', label='Elastic net coefficients',
                   use_line_collection=True)
plt.setp([m, s], color="#2ca02c")
m, s, _ = plt.stem(np.where(lasso.coef_)[0], lasso.coef_[lasso.coef_ != 0],
                   markerfmt='x', label='Lasso coefficients',
                   use_line_collection=True)
plt.setp([m, s], color='#ff7f0e')
plt.stem(np.where(coef)[0], coef[coef != 0], label='true coefficients',
         markerfmt='bx', use_line_collection=True)

plt.legend(loc='best')
plt.title("Lasso $R^2$: %.3f, Elastic Net $R^2$: %.3f"
          % (r2ScoreLasso, r2ScoreElasticNet))
plt.show()