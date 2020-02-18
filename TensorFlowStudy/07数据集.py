import tensorflow

# 兼容性处理
tf = tensorflow.compat.v1
tf.disable_eager_execution()

r = tf.random_normal([10, 3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator() # 创建迭代器
next_row = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)  # 初始化迭代器
    while True:
        try:   # 尝试去取数据
            print(sess.run(next_row))
        except tf.errors.OutOfRangeError:  # 取完时报错，结束运行
            break
# [ 1.4232422  -0.44127613 -0.25749728]
# [0.08910907 1.1862833  1.6656241 ]
# [ 1.1029763  0.271445  -1.5938461]
# [ 0.46730274  1.6858618  -0.8674341 ]
# [ 0.34858897  0.8209042  -1.5865512 ]
# [0.7489046 0.5045604 0.8409652]
# [-0.05139712 -1.4506148  -1.030955  ]
# [ 2.7066352  -0.32011124  0.09240205]
# [-0.04591413  0.3809974   1.0171616 ]
# [ 1.357736    1.4105271  -0.07516529]