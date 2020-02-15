import numpy as np
from matplotlib import pyplot as plt
from scipy import misc, ndimage

face = misc.face()
# 可视化数据
plt.imshow(face)
# 高斯模糊
plt.imshow(ndimage.gaussian_filter(face, sigma=5))
# 旋转变换
plt.imshow(ndimage.rotate(face, 45))
# 卷积操作
k = np.random.randn(2, 2, 3)
plt.imshow(ndimage.convolve(face, k))
plt.show()
