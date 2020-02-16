import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

# 创建 0 阶变量
mammal = tf.Variable('Elephant', tf.string)
print(mammal)
# <tf.Variable 'Variable:0' shape=() dtype=string>

# 创建 1 阶 Tensor 对象
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
print(cool_numbers)
# <tf.Variable 'Variable_1:0' shape=(2,) dtype=float32>

# 创建 2 阶 Tensor 对象
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
print(squarish_squares)
# <tf.Variable 'Variable_2:0' shape=(2, 2) dtype=int32>

# 创建 n 阶 Tensor 对象
imag = tf.zeros([10, 299, 299, 3])
print(imag)
# Tensor("zeros:0", shape=(10, 299, 299, 3), dtype=float32)
# 确定对象的阶
r = tf.rank(imag)
print(r)
# Tensor("Rank:0", shape=(), dtype=int32)
# 使用会话运行获得结果
with tf.Session() as sess:
    print(sess.run(r))
    # 4