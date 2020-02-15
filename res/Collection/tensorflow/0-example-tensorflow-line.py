import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# x = [[80,3,50],[90,8,70],[180,20,120],[140,16,90]]
# x = [[80],[90],[100],[110]]
# y = [[11],[12],[13],[14]]
x = np.array([[10.0*i,] for i in range(100)])
y = np.array([[1.0*i,] for i in range(100)])
# y = [11,12.5,20,18]
# x_pred = [[0,0,0]]
x_pred = np.array([[10.0*i,] for i in range(100)])
# print(x, '\n', x_pred)
# exit(0)



tf_x = tf.placeholder(tf.float32, [None,1])     # input x
tf_y = tf.placeholder(tf.float32, [None,1])     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 20, tf.nn.relu)          # hidden layer:一层隐藏层，有20个神经元
output = tf.layers.dense(l1, 1)                     # output layer

loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph


for step in range(200):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 10 == 0:
        print('loss is: ' + str(l))
        # print('prediction is:' + str(pred))

output_pred = sess.run(output,{tf_x:x_pred})
print('input is:' + str(x_pred[0][:]))
print('output is:' + str(output_pred[0][0]))
# --------------------- 
# 作者：roguesir 
# 来源：CSDN 
# 原文：https://blog.csdn.net/roguesir/article/details/79383122 
# 版权声明：本文为博主原创文章，转载请附上博文链接！
plt.figure()
plt.plot(x[:,0], y[:,0])
plt.plot(x_pred[:,0], output_pred[:,0], 'r*')
plt.show()