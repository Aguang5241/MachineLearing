import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 以下代码用于定义神经网络: 一个输入的输入层->10个神经元的中间层->1个输出的输出层
# 构建神经网络时，对应于实际得输入和输出的占位Tensor
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构建中间层：x*W + b
# 权重
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 偏置值
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 定义操作
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 定义激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)
# 一个中间层可以称为一个块，多层神经网络就是包含多个块的网络

# 构建输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 预测值, prediction 相当于神经网络中，输入从占位Tensor x输入，经过网络传导到输出层的整个流程
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 定义代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
# 定义训练器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 启动会话(session)
# 前面的步骤只会构建图(Graph), 但是实际的操作是不进行操作
with tf.Session() as sess:
    # 初始化变量(Variable)
    sess.run(tf.global_variables_initializer())

    # 进行优化迭代
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 优化完成，使用神经网络进行预测
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 对结果进行可视化
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_data, y_data, "d")
    ax.plot(x_data, prediction_value, "r-", lw=2)
    plt.show()