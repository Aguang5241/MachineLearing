import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

g_l = tf.Graph()
with g_l.as_default():
    a = tf.constant(3.0, name='a')
    b = tf.constant(4.0, name='b')
    c = a + b

    sess = tf.Session()  # tensorflow2.0 版本

    print(sess.run(a))
    # 3.0
    print(sess.run(b))
    # 4.0
    print(sess.run(c))
    # 7.0
