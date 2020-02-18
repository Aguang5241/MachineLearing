import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

# 创建层
x = tf.placeholder(tf.float32, shape=[None, 3]) # 创建一个占位符
linear_model = tf.layers.Dense(units=1) # 创建一个 Dense 层
y = linear_model(x) # 将 x 输入到 Dense 层，然后得到 y
print(x)
# Tensor("Placeholder:0", shape=(None, 3), dtype=float32)
print(y)
# Tensor("dense/BiasAdd:0", shape=(None, 1), dtype=float32)
# 初始化层
init = tf.global_variables_initializer()   # 定义一个全局初始化操作
with tf.Session() as sess:
    sess.run(init)           # 全局初始化操作
    # 执行层
    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))