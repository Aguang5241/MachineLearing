import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

list0 = tf.constant([1, 2, 3], dtype=tf.int32)
print(list0)
# Tensor("Const:0", shape=(3,), dtype=int32)

float_tensor = tf.cast(list0, dtype=tf.float32)
print(float_tensor)
# Tensor("Cast:0", shape=(3,), dtype=float32)

print(float_tensor.dtype)
# <dtype: 'float32'>