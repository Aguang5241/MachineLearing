import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

matrix = tf.constant([[4, 9], [16, 25]], tf.int32)
print(matrix)
# Tensor("Const:0", shape=(2, 2), dtype=int32)
zeros = tf.zeros(matrix.shape[1])
print(zeros)
# Tensor("zeros:0", shape=(2,), dtype=float32)

rank_three_tensor = tf.ones([3, 4, 5])
print(rank_three_tensor)
# Tensor("ones:0", shape=(3, 4, 5), dtype=float32)
matrix = tf.reshape(rank_three_tensor, [6, 10])
print(matrix)
# Tensor("Reshape:0", shape=(6, 10), dtype=float32)
matrixB = tf.reshape(matrix, [3, -1])
print(matrixB)
# Tensor("Reshape_1:0", shape=(3, 20), dtype=float32)
matrixAlt = tf.reshape(matrixB, [4, 3, -1])
print(matrixAlt)
# Tensor("Reshape_2:0", shape=(4, 3, 5), dtype=float32)
yet_another = tf.reshape(matrixAlt, [3, 2, -1])
print(yet_another)
# Tensor("Reshape_3:0", shape=(3, 2, 10), dtype=float32)

with tf.Session() as sess:
    print(sess.run(zeros))
    # [0. 0.]