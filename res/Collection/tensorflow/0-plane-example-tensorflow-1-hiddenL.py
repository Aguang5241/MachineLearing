#encoding=utf-8
import tensorflow as tf
import numpy as np
import datetime as d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# 1.定义添加层函数 
def add_layer(inputs,in_size,out_size,activation_function=None): #添加一个层，输入值，输入尺寸，输出尺寸，以及激励函数，此处None默认为线性的激励函数
    weights_pre = tf.Variable(tf.random_normal([in_size,out_size])) #定义权值矩阵，in_size行，out_size列，随机初始权值
    biases_pre = tf.Variable(tf.zeros([1,out_size])+0.1) #定义一个列表，一行，out_size列，值全部为0.1
    y_pre = tf.matmul(inputs,weights_pre) + biases_pre  #weights_pre*inputs+biases_pre，预测值，未激活
    if activation_function is None:
        outputs = y_pre #结果为线性输出，激活结果
    else:
        outputs = activation_function(y_pre)#激励函数处理
    return outputs

# 2.定义数据集 
# 这里可以定义随机数据，通过神经网络学习来完成线性拟合功能
# --------------------------------linear-----------------------------------
# x_real = np.linspace(-1,1,300)[:,np.newaxis]#[-1,1]之间有300个值，后面[]表示维度，即有300行
# noise = np.random.normal(0,0.05,x_real.shape)#噪声均值为0,方差为0.05,与x_data格式相同
# y_real = x_real-0.5 + noise
# y_real = np.square(x_real)-0.5 + noise
# y_real = np.power(x_real, 3) + np.square(x_real)-0.5 + noise
# —----------------------------plane-------------------------------
#生成一千个点
npoint = 10
input_Multiple = 4
# planeclass = "xe2+xe2-xe3+xe2+b"
planeclass = "xe10+xe5+b"
x_real = np.random.random([npoint,input_Multiple])
x_data = x_real[:,0]
print(x_real.shape)
print(x_real)
#系数矩阵的shape必须是（3，1）。如果是（3，）会导致收敛效果差，猜测可能是y-y_label处形状不匹配
y_raw = np.zeros([npoint,1])
noise = np.random.normal(0, 0.1, y_raw.shape)
# y_real = np.matmul(x_real,[[2],[3]]) + noise         # 多元一次方程
y_raw = 150.0*np.power(x_real[:,0],10)[:,np.newaxis] + 300*np.power(x_real[:,1],5)[:,np.newaxis] + noise    # 二元多次方程
# y_raw = 15.0*np.power(x_real[:,0],2)[:,np.newaxis] + 30*np.power(x_real[:,1],2)[:,np.newaxis] - 45*np.power(x_real[:,2],3)[:,np.newaxis]  + 5*np.power(x_real[:,2],2)[:,np.newaxis] + noise    # 3元多次方程


# 归一化输出值
scaler = StandardScaler()
scaler.fit(y_raw)
y_real = scaler.transform(y_raw)

# print(np.square(x_real[:,0])[:,np.newaxis].shape)
# print('noise: ', noise)
# print('y_real: ', y_real)
# exit(0)

# 定义placeholder接收数据
xs = tf.placeholder(tf.float32,[None,input_Multiple],name='x_input')
ys = tf.placeholder(tf.float32,[None,1],name='y_input')

# 3.定义隐含层，神经元数目为10
hidden_layer_1 = 10
# 隐含层输入层input（1个神经元）：输入1个神经元，隐藏层10个神经元
l1 = add_layer(xs,input_Multiple,hidden_layer_1,activation_function = tf.nn.relu)   # relu
 
# 4.定义输出层，隐含层l1的输出为输出层predicti里写代码片输出层
prediction = add_layer(l1,hidden_layer_1,1,activation_function = None)
# 输出层也是1个神经元

# 5.定义loss函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
# 先求误差平方和的和求平均，reduce_sum表示对矩阵求和，reduction_indices=[1]方向

# 6.选择合适的优化器来优化训练过程的损失函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) 
# 学习效率0.1,要求小于1

# 7.定义存储
saver = tf.train.Saver()

# 8.初始化变量
init = tf.initialize_all_variables()   #初始化所有变量
sess = tf.Session()
sess.run(init)   #激活

# 9.可视化结果
# fig = plt.figure()
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
# ax = fig.add_subplot(1,1,1)
ax.scatter(x_real[:,0], x_real[:,1], y_raw, c='b', label="sample %d points" % npoint)
# plt.ion()        #将画图模式改为交互模式
# plt.show()

iter_number = 10000
print("===========================================")
ans = input("# Train or Not Y/[N]:")
if ans.lower() != "n":
    for i in range(iter_number):
        sess.run(train_step,feed_dict = {xs:x_real,ys:y_real})
        if i%200 == 0 or i == (iter_number-1):
            _, loss_now, prediction_value = sess.run([train_step, loss, prediction],feed_dict={xs:x_real,ys:y_real})
            print('[plane][1 hidden layer:%d neure][%dpoints][%d]loss is: ' % (hidden_layer_1, npoint, i)  + str(loss_now))
            #画出预测
else:
    save_path = saver.restore(sess, "./plane-1-hiddenL/[%s]plane[%s]-1-hiddenL-[%dneure][%diter].ckpt" % (d.datetime.now().strftime("[%H_%M_%S]"), planeclass, hidden_layer_1, iter_number) )
    _, loss_now, prediction_value = sess.run([train_step, loss, prediction],feed_dict={xs:x_real,ys:y_real})
    print('[plane][1 hidden layer:%d neure][%dpoints][%d]loss is: ' % (hidden_layer_1, npoint, iter_number-1)  + str(loss_now))

# 转回输出值

prediction_real = scaler.inverse_transform(prediction_value)

# 存储训练结果
print("===========================================")
ans = input("# Save or Not Y/[N]: ")
if ans.lower() != "n":
    save_path = saver.save(sess, "./plane-1-hiddenL/[%s]plane[%s]-1-hiddenL-[%dneure][%dpoints][%diter].ckpt" % (d.datetime.now().strftime("[%H_%M_%S]"), planeclass, hidden_layer_1, npoint, iter_number) )

# 作图
ax.scatter(x_real[:,0], x_real[:,1], prediction_real, c='r', label="[1-hidden-layer-%dneure][%dpoints] \n predicted data at iter = %d \n loss = " % (hidden_layer_1, npoint, iter_number-1) +  str(loss_now))
handles, labels = plt.gca().get_legend_handles_labels()     # 显示legend
plt.legend(handles, labels, loc = 'best')
# plt.ioff()    # 关闭交互模式，并停留
print("===========================================")
ans = input("# Save figure or Not Y/[N]:")
if ans.lower() != "n":
    plt.savefig('./plane-1-hiddenL-fig/[%s]plane[%s]-example-tensorflow-[1-hidden-layer-%dneure][%dpoints][%diter].png' % (d.datetime.now().strftime("[%H_%M_%S]"), planeclass, hidden_layer_1, npoint, iter_number-1) )

plt.show()
