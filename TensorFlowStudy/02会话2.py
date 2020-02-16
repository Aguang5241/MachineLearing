import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

x = tf.placeholder(tf.int32, shape=[3])  # 创建一个占位符
y = tf.square(x)  # 创建一个操作，对 x 取平方得到 y
with tf.Session() as sess:
    feed = {x: [1, 2, 3]}  # 运行时，给占位符喂的值
    print(sess.run(y, feed_dict=feed))
    # [1 4 9]
    print(sess.run(y, {x: [4, 5, 6]}))
    # [16 25 36]
