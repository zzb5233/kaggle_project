# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import LeNet5_infernece
import os
import numpy as np
import pandas as pd
BATCH_SIZE = 100
LEARNING_RATE_BASE     = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1001
MOVING_AVERAGE_DECAY = 0.99
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE  = 'test.csv'
VALIDATION_NUMBER  = 1000
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"
PREDICTION_FILE = './conv_submission.csv'

#data preprocessing
def extract_images_and_labels(dataset, validation = False):
    images = dataset[:, 1:].reshape(-1, 28, 28, 1)

    labels_dense = dataset[:, 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * LeNet5_infernece.NUM_LABELS
    labels_one_hot = np.zeros((num_labels, LeNet5_infernece.NUM_LABELS))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    if validation:
        num_images = images.shape[0]
        divider = num_images - VALIDATION_NUMBER
        return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
    else:
        return images, labels_one_hot
    
def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]

def pre_process_data():
    train_data = pd.read_csv(TRAIN_DATA_FILE).as_matrix().astype(np.uint8)
    test_data = pd.read_csv(TEST_DATA_FILE).as_matrix().astype(np.uint8)
    
    train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
    
    train = DataSet(train_images, train_labels, dtype = np.float32, reshape = True)
    validation = DataSet(val_images, val_labels, dtype = np.float32, reshape = True)
    
    train_number = train_images.shape[0]
    
    return train, validation, test_data, train_number

def train(train_data, train_number):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_number / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            ts = train_data.next_batch(BATCH_SIZE)
            xs, ys = ts[0], ts[1]
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                
                
#predict data
def predict_data(validation_data, test_data):

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])  
    keep_prob = tf.placeholder(tf.float32)    

    y_hat = model(x, keep_prob)
    batchs = get_batchs(test_data, BATCH_SIZE)
    conv_y_predict = []
    prediction = tf.argmax(y_hat, 1)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            validation_accuracy = sess.run(accuracy, feed_dict={x: validation_data.images, y_: validation_data.labels, keep_prob: 1.0})
            print("validation accuracy", validation_accuracy)            
            for test_image in batchs:
                test_labels = sess.run(prediction, feed_dict={x: test_image, keep_prob: 1.0})
                for label in test_labels:
                    conv_y_predict = np.append(conv_y_predict, label)
            conv_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': np.int32(conv_y_predict)})
            conv_submission.to_csv(PREDICTION_FILE, index = False)              
        else:
            print('No checkpoint file found')
            return    
                
                
def main(argv=None):
    train_data, validation_data, test_data, train_number = pre_process_data()
    train(train_data, train_number)
    predict_data(validation_data, test_data)

if __name__ == '__main__':
    main()