import scipy

####################1.版本查看######################
print(scipy.__version__)
# 1.4.1
####################2.常量模块######################
from scipy import constants
# 圆周率
print(constants.pi)
# 3.141592653589793
# 黄金分割常数
print(constants.golden)
# 1.618033988749895
# 光速
print(constants.c)
# 等同于 print(constants.speed_of_light)
# 299792458.0
# 普朗克系数
print(constants.h)
# 等同于 print(constants.Planck)
# 6.62607015e-34
