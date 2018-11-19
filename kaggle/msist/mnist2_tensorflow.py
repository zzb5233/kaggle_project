# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd

#1 加载数据集
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#2 把图片数据取出来，进行处理
x_train = train.iloc[:,1:].values
x_train = x_train.astype(np.float)
x_test = test.iloc[:,:].values
x_test = x_test.astype(np.float)

#3 给到的图片灰度值在0-255，这里将图片的信息控制在0~1之间
x_train = np.multiply(x_train, 1.0/255)
x_test = np.multiply(x_test, 1.0/255)

#4 计算图片的长和高，下面会用到
image_size = x_train.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)


#5 把数据集的标签结果取出来
labels_train = train.iloc[:,0].values
label_count = np.unique(labels_train).shape[0]

#写一个对Label进行one-hot处理的函数
def dense_to_ont_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

#6 对Label进行one-hot处理
labels = dense_to_ont_hot(labels_train, label_count)
labels = labels.astype(np.uint8)

#7 设置批次大小，求得批次量
batch_size = 128
n_batch = int(len(x_train)/batch_size)

#8 定义两个placeholder,用来承载数据，因为每个图片都是一个784维数据，所以x是784列；
#  因为要把图片识别为0-9的10个数字，也就是10个标签，所以y是10列
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#9 定义几个处理函数
def weight_variable(shape):
    #初始化权重，正态分布，标准方差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    #初始化偏置值，设置非零避免死神经元
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x,w):
    #卷积不改变输入的shape
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#对Tensorflow的池化进行封装
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#10 把输入变换成一个4d的张量，第二三个对应的是图片的长和宽，第四个参数是颜色
x_image = tf.reshape(x,[-1,28,28,1])

#11 计算32个特征，每3*3patch，第一二个参数指的是patch的size,第三个参数是输入的
#   channelss,第四个参数是输出的channels
W_conv1 = weight_variable([3,3,1,32])

#12 偏差的shape应该和输出的shape一致
b_conv1 = bias_variable([32])

#28*28的图片卷积时步长为1，随意卷积后大小不变，按2*2最大值池化，相当于从2*2块中提取一个最大值
#所以池化后大小为[28/2,28/2]=[14,14],第二次池化后为[7,7]

#13 对数据进行卷积操作
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

#14 对结果做池化，max_pool_2x2之后，图片变成14*14
h_pool1 = max_pool_2x2(h_conv1)

#15 在以前的基础上，生成了64个特征
W_conv2 = weight_variable([6,6,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#16 max_pool_2x2之后，图片变成7*7
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])

#17 构造一个全连接的神经网络，1024个神经元
w_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#18 做Droupout操作
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#19 把1024个神经元的输入变为一个10维输出
w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

predictions = tf.nn.softmax(y_conv)
#20 创建损失函数，以交叉熵的平均值为衡量
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv))

#21 用梯度下降法优化参数
train_step_1 = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

#22 计算准确度
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#23 设置保存模型的文件名参数
global_step = tf.Variable(0,name='global_step',trainable=False)
#saver = tf.train.Saver()

#24 初始化变量
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    #25 初始化
    sess.run(init)

    #这是载入以前训练好的模型的语句，有需要采用，注意把文件名改成成绩比较好的周期
    #saver.restore(sess,'model.ckpt-12')

    #迭代20周期
    for epoch in range(20):
        print('epoch',epoch+1)
        for batch in range(n_batch):
            #27 每次取出一个数据进行训练
            batch_x = x_train[(batch)*batch_size:(batch+1)*batch_size]
            batch_y = labels[(batch)*batch_size:(batch+1)*batch_size]

            #28 [重要] 这是最终运行整个训练模型的语句
            sess.run(train_step_1,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})

        batch_x = x_train[n_batch*batch_size:]
        batch_y = labels[n_batch*batch_size:]

        #28 [重要] 这是最终运行整个训练模型的语句
        sess.run(train_step_1,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})

    #保存训练模型
    saver.save(sess,'ModelconvNN2/model.ckpt')

with tf.Session() as sess1: 
    saver.restore(sess1,'ModelconvNN2/model.ckpt')
    #29 计算预测
    #一次预测28000个内存不足，故分批预测
    test_batch_size = 4000 
    test_n_batch = int(len(x_test)/test_batch_size)      
    for batch in range(test_n_batch):
        filename = 'ConvNN'+str(batch)+'.csv'
        test_batch_x = x_test[batch*test_batch_size:(batch+1)*test_batch_size]
        myPrediction = sess1.run(predictions,feed_dict={x:test_batch_x,keep_prob:1.0})
        label_test = np.argmax(myPrediction,axis=1)
        pd.DataFrame(label_test).to_csv(filename)
