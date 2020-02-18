import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

init = tf.global_variables_initializer()

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)   # 创建优化器
train = optimizer.minimize(loss)    # 优化损失

with tf.Session() as sess:
    sess.run(init)  # 全局初始化
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        if i % 10 == 0:
            print(loss_value)

# 32.916454
# 1.2605368
# 0.41772225
# 0.37350133
# 0.35124654
# 0.3307888
# 0.31153488
# 0.29340187
# 0.27632436
# 0.26024085