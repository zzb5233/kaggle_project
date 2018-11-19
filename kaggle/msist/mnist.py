import pandas as pd
import tensorflow as tf
import skflow
import numpy as np
import sys

train = pd.read_csv('../Datasets/MNIST/train.csv')
test = pd.read_csv('../Datasets/MNIST/test.csv')
y_train = train['label']
X_train = train.drop('label', 1)



classifier = skflow.TensorFlowLinearClassifier(n_classes=10, batch_size=100, steps=1000, learning_rate=0.01)
classifier.fit(X_train, y_train)
linear_y_predict = classifier.predict(X_test)
linear_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': linear_y_predict})
linear_submission.to_csv('../Datasets/MNIST/linear_submission.csv', index = False)



classifier = skflow.TensorFlowDNNClassifier(hidden_units=[200, 50, 10], n_classes = 10, steps=5000, learning_rate=0.01, batch_size=50)
classifier.fit(X_train, y_train)
dnn_y_predict = classifier.predict(X_test)
dnn_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': dnn_y_predict})
dnn_submission.to_csv('../Datasets/MNIST/dnn_submission.csv', index = False)




def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_model(X, y):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
        
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, keep_prob=0.5)
    return skflow.models.logistic_regression(h_fc1, y)

classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=10, batch_size=100, steps=20000, learning_rate=0.001)
classifier.fit(X_train, y_train)

conv_y_predict = []

for i in np.arange(100, 28001, 100):
    conv_y_predict = np.append(conv_y_predict, classifier.predict(X_test[i - 100:i]))
    
conv_submission = pd.DataFrame({'ImageId':range(1, 28001), 'Label': np.int32(conv_y_predict)})
conv_submission.to_csv('../Datasets/MNIST/conv_submission.csv', index = False)