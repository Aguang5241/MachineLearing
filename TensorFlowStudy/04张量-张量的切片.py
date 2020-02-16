import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

matrix = tf.constant([[4, 9], [16, 25]], tf.int32)
scalar = matrix[1, 0]

with tf.Session() as sess:
    print(sess.run(scalar))
    # 16