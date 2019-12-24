#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1.定义添加层函数 
def add_layer(inputs,in_size,out_size,activation_function=None): #添加一个层，输入值，输入尺寸，输出尺寸，以及激励函数，此处None默认为线性的激励函数
    w_pre = tf.Variable(tf.random_normal([in_size,out_size])) #定义权值矩阵，in_size行，out_size列，随机初始权值
    b_pre = tf.Variable(tf.zeros([1,out_size])+0.1) #定义一个列表，一行，out_size列，值全部为0.1
    y_pre = tf.matmul(inputs,w_pre)+b_pre  #w_pre*inputs+b_pre，预测值，未激活
    if activation_function is None:
        outputs = y_pre #结果为线性输出，激活结果
    else:
        outputs = activation_function(y_pre)#激励函数处理
    return outputs

# 2.定义数据集 
# 这里可以定义随机数据，通过神经网络学习来完成线性拟合功能
x_real = np.linspace(-1,1,300)[:,np.newaxis]#[-1,1]之间有300个值，后面[]表示维度，即有300行
noise = np.random.normal(0,0.05,x_real.shape)#噪声均值为0,方差为0.05,与x_data格式相同
# y_real = np.square(x_real)-0.5 + noise
y_real = np.power(x_real, 3) + np.square(x_real)-0.5 + noise
# 定义placeholder接收数据
xs = tf.placeholder(tf.float32,[None,1],name='x_input')
ys = tf.placeholder(tf.float32,[None,1],name='y_input')

# 3.定义隐含层:2层
hidden_layer_1 = 10
hidden_layer_2 = 10
# 隐含层输入层input（1个神经元）：输入1个神经元，隐藏层10个神经元
l1 = add_layer(xs,1,hidden_layer_1,activation_function = tf.nn.relu)
# 隐含层输入层input（1个神经元）：输入10个神经元，隐藏层5个神经元
l2 = add_layer(l1,hidden_layer_1,hidden_layer_2,activation_function = tf.nn.relu)

# 4.定义输出层，隐含层l2的输出为输出层predicti里写代码中的输入
prediction = add_layer(l2,hidden_layer_2,1,activation_function = None)
# 输出层也是1个神经元

# 5.定义loss函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
# 先求误差平方和的和求平均，reduce_sum表示对矩阵求和，reduction_indices=[1]方向

# 6.选择合适的优化器来优化训练过程的损失函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) 
# 学习效率0.1,要求小于1

# 7.初始化变量
init = tf.initialize_all_variables()#初始化所有变量

sess = tf.Session()
sess.run(init)#激活

# 8.可视化结果
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_real,y_real)
plt.ion()
plt.show()

iter_number = 1000
for i in range(iter_number):
    sess.run(train_step,feed_dict = {xs:x_real,ys:y_real})
    if i%50 == 0 or i == (iter_number-1):
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        _, loss_now, prediction_value = sess.run([train_step, loss, prediction],feed_dict={xs:x_real,ys:y_real})
        print('[2 hidden layer: %d x %d neure][%d]loss is: ' % (hidden_layer_1, hidden_layer_2, i)  + str(loss_now))
        #画出预测
        lines = ax.plot(x_real,prediction_value,'r-',lw=2)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels = [ '[2-hidden-layer-%dx%dneure] \n predicted data at iter = %d \n loss = ' % (hidden_layer_1, hidden_layer_2, i) +  str(loss_now), 'origin data'], loc = 'best')
        plt.pause(1)
    if i == (iter_number-1):   # 
        plt.savefig('example-tensorflow-[2-hidden-layer-%dx%dneure][%diter].png' % (hidden_layer_1, hidden_layer_2, i) )

