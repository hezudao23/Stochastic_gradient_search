# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:06:40 2018

@author: yunlo
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
Model_num = 2 # test a small one to have it work

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1_dict = {}
b_conv1_dict = {}
W_conv2_dict = {}
b_conv2_dict = {}
W_fc1_dict = {}
b_fc1_dict = {}
W_fc2_dict = {}
b_fc2_dict = {}
y_conv_dict = {}
cross_entropy_dict = {}
grad_W_conv1_dict = {}
grad_b_conv1_dict = {}
grad_W_conv2_dict = {}
grad_b_conv2_dict = {}
grad_W_fc1_dict = {}
grad_b_fc1_dict = {}
grad_W_fc2_dict = {}
grad_b_fc2_dict = {}
new_W_conv1_dict = {}
new_b_conv1_dict = {}
new_W_conv2_dict = {}
new_b_conv2_dict = {}
new_W_fc1_dict = {}
new_b_fc1_dict = {}
new_W_fc2_dict = {}
new_b_fc2_dict = {}
correct_prediction_dict = {}
accuracy_dict = {}

learning_rate = 1e-4

beta = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

for i in range(Model_num):
    """ Step 1 To build the model"""
    name_str = 'model_' + str(i)
    W_conv1_dict[name_str] = weight_variable([5, 5, 1, 32])
    b_conv1_dict[name_str] = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1_dict[name_str]) + b_conv1_dict[name_str])
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2_dict[name_str] = weight_variable([5, 5, 32, 64])
    b_conv2_dict[name_str] = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2_dict[name_str]) + b_conv2_dict[name_str])
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1_dict[name_str] = weight_variable([7 * 7 * 64, 1024])
    b_fc1_dict[name_str] = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1_dict[name_str]) + b_fc1_dict[name_str])
    W_fc2_dict[name_str] = weight_variable([1024, 10])
    b_fc2_dict[name_str] = bias_variable([10])
    y_conv_dict[name_str] = tf.matmul(h_fc1, W_fc2_dict[name_str]) + b_fc2_dict[name_str]
    """ Step 2 To compute the loss """
    cross_entropy_dict[name_str] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, \
                      logits=y_conv_dict[name_str]))
    """ Step 3 To find the gradient """
    [grad_W_conv1_dict[name_str], grad_b_conv1_dict[name_str], grad_W_conv2_dict[name_str],  grad_b_conv2_dict[name_str],\
     grad_W_fc1_dict[name_str],  grad_b_fc1_dict[name_str], grad_W_fc2_dict[name_str], grad_b_fc2_dict[name_str]] =\
     tf.gradients(ys = cross_entropy_dict[name_str], \
                  xs = [W_conv1_dict[name_str], b_conv1_dict[name_str], W_conv2_dict[name_str],  b_conv2_dict[name_str],\
     W_fc1_dict[name_str],  b_fc1_dict[name_str], W_fc2_dict[name_str], b_fc2_dict[name_str]])
    
    """Along the gradient to update these parameters """
    new_W_conv1_dict[name_str] = W_conv1_dict[name_str].assign(W_conv1_dict[name_str] - \
                    learning_rate * grad_W_conv1_dict[name_str] + tf.sqrt(2 * learning_rate/beta) * np.random.rand())
    new_b_conv1_dict[name_str] = b_conv1_dict[name_str].assign(b_conv1_dict[name_str] - \
                    learning_rate * grad_b_conv1_dict[name_str] + tf.sqrt(2 * learning_rate/beta) * np.random.rand())
    new_W_conv2_dict[name_str] = W_conv2_dict[name_str].assign(W_conv2_dict[name_str] - \
                    learning_rate * grad_W_conv2_dict[name_str] + tf.sqrt(2 * learning_rate/beta) * np.random.randn())
    new_b_conv2_dict[name_str] = b_conv2_dict[name_str].assign(b_conv2_dict[name_str] - \
                    learning_rate * grad_b_conv2_dict[name_str] +tf.sqrt(2 * learning_rate/beta) * np.random.randn())
    new_W_fc1_dict[name_str] = W_fc1_dict[name_str].assign(W_fc1_dict[name_str] - \
                  learning_rate * grad_W_fc1_dict[name_str] + tf.sqrt(2 * learning_rate/beta) * np.random.randn())
    new_b_fc1_dict[name_str] = b_fc1_dict[name_str].assign(b_fc1_dict[name_str] - \
                  learning_rate * grad_b_fc1_dict[name_str] +  tf.sqrt(2 * learning_rate/beta) * np.random.randn())
    new_W_fc2_dict[name_str] = W_fc2_dict[name_str].assign(W_fc2_dict[name_str] - \
                  learning_rate * grad_W_fc2_dict[name_str] +  tf.sqrt(2 * learning_rate/beta) * np.random.randn())
    new_b_fc2_dict[name_str] = b_fc2_dict[name_str].assign(b_fc2_dict[name_str] - \
                  learning_rate * grad_b_fc2_dict[name_str] +  tf.sqrt(2 * learning_rate/beta) * np.random.randn())
    
    correct_prediction_dict[name_str] = tf.equal(tf.argmax(y_conv_dict[name_str], 1), tf.argmax(y_, 1))
    accuracy_dict[name_str] = tf.reduce_mean(tf.cast(correct_prediction_dict[name_str], tf.float32))

"""Starting from here, we would like to update the beta"""
    
beta_var = tf.Variable([0.1], dtype = tf.float32)
beta_dot_var = tf.Variable([0.1 * 0.1], dtype = tf.float32)    

beta_dot_holder = tf.placeholder(tf.float32)
beta_2dot_holder = tf.placeholder(tf.float32)   

#def avg_sq_W_conv1(W_conv1_dict):
#    return sum(np.square( list(W_conv1_dict.values()) ))


#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  beta_value = 100
  
  for i in range(1000):
    batch = mnist.train.next_batch(50)
    W_result = sess.run(W_conv1_dict)
           
    for j in range(Model_num):
        name_str = 'model_' + str(j)   
        _, _, _, _, _, _, _, _, c = sess.run([new_W_conv1_dict[name_str], new_b_conv1_dict[name_str],\
                                             new_W_conv2_dict[name_str], new_b_conv2_dict[name_str],\
                                             new_W_fc1_dict[name_str], new_b_fc1_dict[name_str],\
                                             new_W_fc2_dict[name_str], new_b_fc2_dict[name_str], cross_entropy_dict[name_str]], \
                                            feed_dict = {x:batch[0], y_:batch[1], beta : beta_value})   
        if i % 100 == 0:
            train_accuracy = accuracy_dict[name_str].eval(feed_dict={
                    x: batch[0], y_: batch[1]})
            print('step %d, training accuracy %f' % (i, train_accuracy))
           

  #print('test accuracy %f' % accuracy.eval(feed_dict={
  #    x: mnist.test.images, y_: mnist.test.labels}))
  #b_fc2_result = sess.run(b_fc2)
  #print(b_fc2_result)