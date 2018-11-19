# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE  = 'test.csv'
BATCH_SIZE      = 100
TRAINING_STEPS  = 20000
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"
PREDICTION_FILE = '../Datasets/MNIST/conv_submission.csv'


#data preprocessing
def extract_images_and_labels(dataset, validation = False):
    images = dataset[:, 1:].reshape(-1, 28, 28, 1)

    labels_dense = dataset[:, 0]
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * 10
    labels_one_hot = np.zeros((num_labels, 10))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    if validation:
        num_images = images.shape[0]
        divider = num_images - 1000
        return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
    else:
        return images, labels_one_hot

def extract_images(dataset):
    return dataset.reshape(-1, 28*28)


train_data = pd.read_csv(TRAIN_DATA_FILE).as_matrix().astype(np.uint8)
test_data = pd.read_csv(TEST_DATA_FILE).as_matrix().astype(np.uint8)

train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_images = extract_images(test_data)

train = DataSet(train_images, train_labels, dtype = np.float32, reshape = True)
validation = DataSet(val_images, val_labels, dtype = np.float32, reshape = True)
test = test_images

 
#build model
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]

def model(input_tensor, keep_prob):
    with tf.variable_scope('layer1'):
        weights = weight_variable([5,5,1,32])
        biases = bias_variable([32])
        x_image = tf.reshape(input_tensor, [-1,28,28,1])
        layer1 = tf.nn.relu(conv2d(x_image, weights) + biases)
        pool1 = max_pool_2x2(layer1)

    with tf.variable_scope('layer2'):
        weights = weight_variable([5,5,32,64])
        biases = bias_variable([64])
        layer2 = tf.nn.relu(conv2d(pool1, weights) + biases)
        pool2 = max_pool_2x2(layer2)
        
    with tf.variable_scope('layer3'):
        weights = weight_variable([7*7*64, 1024])
        biases = bias_variable([1024])
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
        layer3 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)
        drop3 = tf.nn.dropout(layer3, keep_prob)
    
    with tf.variable_scope('layer3'):
        weights = weight_variable([1024, 10])
        biases = bias_variable([10])
        y_hat = tf.matmul(drop3, weights) + biases    
    return y_hat

#train model
def train_model():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])  
    keep_prob = tf.placeholder(tf.float32)
    
    y_hat = model(x, keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_hat))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()   
        
        for i in range(TRAINING_STEPS):
            batch = train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],  keep_prob: 1.0})
            if i % 1000 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1],  keep_prob: 1.0})
                print("After %d training step(s), train accuracy on training batch is %g." % (i, train_accuracy))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        validation_accuracy = sess.run(accuracy, feed_dict={x: validation.images, y_: validation.labels, keep_prob: 1.0})
        print("validation accuracy", validation_accuracy)


#predict data
def predict_data():

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])  
    keep_prob = tf.placeholder(tf.float32)    

    y_hat = model(x, keep_prob)
    batchs = get_batchs(test, BATCH_SIZE)
    conv_y_predict = []
    prediction = tf.argmax(y_hat, 1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for test_image in batchs:
                test_labels = sess.run(prediction, feed_dict={x: test_image, keep_prob: 1.0})
                for label in test_labels:
                    conv_y_predict = np.append(conv_y_predict, label)
            conv_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': np.int32(conv_y_predict)})
            conv_submission.to_csv(PREDICTION_FILE, index = False)              
        else:
            print('No checkpoint file found')
            return    
       
if __name__ == '__main__':
    train_model()
    predict_data()