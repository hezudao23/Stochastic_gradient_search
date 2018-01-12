# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:41:48 2018

@author: yunlo
"""
import tensorflow as tf
import numpy as np
#from joblib import Parallel, delayed
#import multiprocessing
# Model parameters
Model_num = 20
W_dict = {}
b_dict = {}
model_dict = {}
loss_dict= {}
grad_W_dict = {}
grad_b_dict = {}
new_W_dict = {}
new_b_dict = {}
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
learning_rate = 0.001

#beta_dot_grad = tf.Variable
#beta_grad = tf.Variable
beta = tf.placeholder(tf.float32)

for i in range(Model_num):
    W_dict['model_' + str(i)] = tf.Variable([.3], dtype=tf.float32)
    b_dict['model_' + str(i)] = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
    model_dict['model_' + str(i)] = tf.multiply(W_dict['model_'+str(i)], x) + b_dict['model_' + str(i)]

# loss
    loss_dict['model_'+str(i)] = tf.reduce_sum(tf.square(model_dict['model_'+str(i)] - y)) # sum of the squares
# optimizer
    [grad_W_dict['model_'+str(i)], grad_b_dict['model_'+str(i)]] = \
    tf.gradients(ys = loss_dict['model_'+str(i)], xs = [W_dict['model_'+str(i)], b_dict['model_'+str(i)]])
    new_W_dict['model_'+str(i)] = W_dict['model_'+str(i)].assign(W_dict['model_'+str(i)] -learning_rate * grad_W_dict['model_'+str(i)]\
              + tf.sqrt(2 * learning_rate/beta) * np.random.rand())
    new_b_dict['model_'+str(i)] = b_dict['model_'+str(i)].assign(b_dict['model_'+str(i)] -learning_rate * grad_b_dict['model_'+str(i)]\
              + tf.sqrt(2 * learning_rate/beta) * np.random.rand())

""" Here we would like to initialize beta """
beta_var = tf.Variable([0.1], dtype = tf.float32)
beta_dot_var = tf.Variable([0.1 * 0.1], dtype = tf.float32)    

beta_dot_holder = tf.placeholder(tf.float32)
beta_2dot_holder = tf.placeholder(tf.float32)

"""Update beta_t and beta_dot_t"""


new_beta_var = beta_var.assign(beta_var + learning_rate * beta_dot_holder)
new_beta_dot_var = beta_dot_var.assign(beta_dot_var + learning_rate * beta_2dot_holder)
"""I test the exponential varying"""

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
"""This is the function to compute the avergae of <W^2> """
def average_square_W(W_dict): 
    return sum(np.square(list(W_dict.values())))/len(W_dict)
"""This is the function to compute the avergae of <b^2> """
def average_square_b(b_dict):
    return sum(np.square(list(b_dict.values())))/len(b_dict)
"""This is the function to compute derivative of <W^2> """
def der_average_square_W(W_dict, grad_W_dict, beta_value):
    return -np.dot(np.squeeze(list(W_dict.values())), np.squeeze(list(grad_W_dict.values())))/ len(W_dict) + 2 /beta_value
"""This is the function to compute derivative of <b^2> """
def der_average_square_b(b_dict, grad_b_dict, beta_value):
    return -np.dot(np.squeeze(list(b_dict.values())), np.squeeze(list(grad_b_dict.values()))) / len(b_dict) + 2 /beta_value

"""This is the function to compute the average of square gradient W """
def average_square_grad_W(grad_W_dict):
    return np.sum( np.square( np.squeeze( list( grad_W_dict.values()) ) ) ) / len(grad_W_dict)

"""This is the function to compute the average of square graidnet b"""
def average_square_grad_b(grad_b_dict):
    return np.sum( np.square( np.squeeze( list( grad_b_dict.values()) ) ) ) / len(grad_b_dict)
"""Here is the key part to compute the average of square gradient (W, b) of log p  """
def average_square_grad_W_and_b_log_p(W_dict, b_dict, W_dict_mem, b_dict_mem, grad_W_dict_mem, grad_b_dict_mem, beta_value):
    res = np.array(list(W_dict.values()))
    W_t_1 = np.repeat(res, Model_num, axis = 1)
    res = np.array(list(b_dict.values()))
    b_t_1 = np.repeat(res, Model_num, axis = 1)
    res = np.array(list(W_dict_mem.values()))
    W_t = np.transpose(np.repeat(res, Model_num, axis = 1))
    res = np.array(list(b_dict_mem.values()))
    b_t = np.transpose(np.repeat(res, Model_num, axis = 1))
    res = np.array(list(grad_W_dict_mem.values()))
    U_W_t = np.transpose(np.repeat(res, Model_num, axis = 1))
    res = np.array(list(grad_b_dict_mem.values()))
    U_b_t = np.transpose(np.repeat(res, Model_num, axis = 1))
    res_W = np.subtract(W_t_1, np.subtract(W_t, np.multiply(U_W_t, learning_rate)))
    res_b = np.subtract(b_t_1, np.subtract(b_t, np.multiply(U_b_t, learning_rate)))
    res_W_exp = np.exp( np.multiply(np.square(res_W), -beta_value/(2 * learning_rate)))
    res_b_exp = np.exp( np.multiply(np.square(res_b), -beta_value/(2 * learning_rate)))
    res_W_exp_D = np.sum(res_W_exp, axis = 1)  # This is the denominator for res_W_exp
    res_b_exp_D = np.sum(res_b_exp, axis = 1) # This is the denominator for res_b_exp 
    res_W_exp_D = res_W_exp_D.reshape(Model_num, 1)
    res_b_exp_D = res_b_exp_D.reshape(Model_num, 1)
    W_Unnormal_weighted = np.multiply(np.multiply(res_W, -beta_value/learning_rate), res_W_exp)
    b_Unnormal_weighted = np.multiply(np.multiply(res_W, -beta_value/learning_rate), res_b_exp)
    W_weighted = np.sum(np.divide(W_Unnormal_weighted, res_W_exp_D), axis = 1)
    b_weighted = np.sum(np.divide(b_Unnormal_weighted, res_b_exp_D), axis = 1)
    
    return np.mean(np.square(W_weighted)) + np.mean(np.square(b_weighted))
        
    
def update_beta_2dot(beta_value, beta_dot_value, average_square_sum, der_average_square_sum,\
                     average_square_grad_sum,  average_square_grad_log_p):
    candidate = np.square(beta_dot_value)/(2 * beta_value) - beta_dot_value * der_average_square_sum / average_square_sum + \
                        2 * beta_value * (average_square_grad_sum - 1/np.square(beta_value) * average_square_grad_log_p) / average_square_sum
    if candidate * learning_rate + beta_dot_value  <= 0: #or candidate * learning_rate + beta_dot_value > 0.1 * beta_value:
        return (0.1 * beta_value - beta_dot_value)/learning_rate
    else:
        return candidate
    #return 0.1 * beta_dot_value

"""This is the test for the parallel for loop in python for our inside for loop"""
#def processInput(i, x_train, y_train, beta_value):
#    _, _, c = sess.run([new_W_dict['model_'+str(i)], new_b_dict['model_'+str(i)] , loss_dict['model_'+str(i)]],\
#                               feed_dict={x: x_train, y: y_train, beta: beta_value})
#    return c

with tf.Session() as sess:
    sess.run(init)

    for j in range(300000):
       
        beta_value = sess.run(beta_var)
        W_dict_mem = sess.run(W_dict) # mem indicates memory
        b_dict_mem = sess.run(b_dict)
        W_grad_dict_mem = sess.run(grad_W_dict, feed_dict = {x : x_train, y : y_train})
        b_grad_dict_mem = sess.run(grad_b_dict, feed_dict = {x : x_train, y : y_train})
       # print(W_dict_mem)
       # print(W_grad_dict_mem)
        #beta_value = 10000
        
        #[b, b_dot] = sess.run([new_beta, new_beta_dot], feed_dict= {})
        
        for i in range(Model_num):
            _, _, c = sess.run([new_W_dict['model_'+str(i)], new_b_dict['model_'+str(i)] , loss_dict['model_'+str(i)]],\
                               feed_dict={x: x_train, y: y_train, beta: beta_value})
            if j % 5000 == 0:
                print(c)
        #c_s = Parallel(n_jobs = multiprocessing.cpu_count)(delayed(processInput)(i, x_train, y_train, beta_value) for i in range(Model_num))
        """ Based on our minimize dissipation result, we would like to use E-L equation to update beta and beta_dot"""
        W_dict_result = sess.run(W_dict)
        b_dict_result = sess.run(b_dict)
        W_grad_dict_result = sess.run(grad_W_dict, feed_dict = {x: x_train, y: y_train})
        b_grad_dict_result = sess.run(grad_b_dict, feed_dict = {x: x_train, y: y_train})
        
        #print(W_dict_result)
        #print(W_grad_dict_result)
        
        average_W_square_res = average_square_W(W_dict_result)
        average_b_square_res = average_square_b(b_dict_result)
        average_square_sum = average_W_square_res + average_b_square_res
        
        der_average_square_W_res = der_average_square_W(W_dict_result, W_grad_dict_result, beta_value)
        der_average_square_b_res = der_average_square_b(b_dict_result, b_grad_dict_result, beta_value)
        der_average_square_sum = der_average_square_W_res + der_average_square_b_res
        
        average_square_grad_W_res = average_square_grad_W(W_grad_dict_result)
        average_square_grad_b_res = average_square_grad_b(b_grad_dict_result)
        average_square_grad_sum = average_square_grad_W_res + average_square_grad_W_res
        
        average_square_grad_log_p =   average_square_grad_W_and_b_log_p(W_dict_result, b_dict_result,\
                                                                        W_dict_mem, b_dict_mem, W_grad_dict_mem, b_grad_dict_mem, beta_value)
        #print(average_square_grad_W_res)
        #print(average_square_grad_b_res)
       
        
        beta_dot_value = sess.run(beta_dot_var)
        beta_2dot_value = update_beta_2dot(beta_value, beta_dot_value, average_square_sum, der_average_square_sum,\
                                           average_square_grad_sum, average_square_grad_log_p)
        
        
        #b = sess.run(new_beta_t, feed_dict = {beta_dot_u : beta_dot_value, beta_2dot_u : beta_2dot_value})
        [b, b_dot] = sess.run([new_beta_var, new_beta_dot_var], feed_dict = {beta_dot_holder : beta_dot_value, beta_2dot_holder : beta_2dot_value})
      
        if j % 5000 == 0:
            print(j)
            print(b)
            print(b_dot)
            print(average_square_grad_sum)
            print(average_square_grad_log_p)
        #beta_dot_t = update_beta_dot(beta_t, beta_dot_t)
        #print(beta_dot_t)
#        value = sess.run(beta)
#        value_1 = sess.run(beta_dot)
        
    # Test model
  
    # Calculate accuracy for 3000 examples
  
    
