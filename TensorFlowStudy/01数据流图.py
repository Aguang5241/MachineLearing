import tensorflow as tf

# 构建
g_l = tf.Graph()
with g_l.as_default():
    a = tf.constant(3.0, name='a')
    b = tf.constant(4.0, name='b')
    c = a + b

print(a)
# Tensor("a:0", shape=(), dtype=float32)
print(b)
# Tensor("b:0", shape=(), dtype=float32)
print(c)
# Tensor("add:0", shape=(), dtype=float32)

# 命名空间
e_0 = tf.constant(0, name="e")
print(e_0)
# tf.Tensor(0, shape=(), dtype=int32)
e_1 = tf.constant(2, name="e")
print(e_1)
# tf.Tensor(2, shape=(), dtype=int32)
with tf.name_scope("outer"):  # 在命名空间 outer 下创建常量
    e_2 = tf.constant(2, name="e")
    print(e_2)
    # tf.Tensor(2, shape=(), dtype=int32)
    with tf.name_scope("inner"):  # 在命名空间 inter 下创建常量
        e_3 = tf.constant(3, name="e")
        print(e_3)
        # tf.Tensor(3, shape=(), dtype=int32)
        e_4 = tf.constant(4, name="e")
        print(e_4)
        # tf.Tensor(4, shape=(), dtype=int32)
    with tf.name_scope("inner"):
        e_5 = tf.constant(5, name="e")
        print(e_5)
        # tf.Tensor(5, shape=(), dtype=int32)