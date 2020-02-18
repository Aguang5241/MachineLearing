import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

# 层函数的快捷方式
x = tf.placeholder(tf.float32, shape=[None, 3])  # 创建一个占位符 x
y = tf.layers.dense(x, units=1)    # 创建一个dense层，输入 x 得到 y

init = tf.global_variables_initializer()  # 定义一个全局初始化操作
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))