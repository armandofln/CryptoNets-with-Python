
# INFERENZA con numpy e numeri interi
# esegue solo un salvataggio dei numpy ricavati da tensorflow

import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#from math import floor
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001


x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

paddings = tf.constant([[0, 0], [1, 0,], [1, 0]])
input_layer = tf.reshape(x, [-1, 28, 28])
input_layer = tf.pad(input_layer, paddings, "CONSTANT") 
input_layer = tf.reshape(input_layer, [-1, 29, 29, 1])

# Input Tensor Shape: [batch_size, 29, 29, 1]
# Output Tensor Shape: [batch_size, 13, 13, 5]

conv = tf.layers.conv2d(
    inputs=input_layer,
    filters=5,
    kernel_size=[5, 5],
    strides=[2, 2],
    padding="valid",
    activation=None,
    name='convolution')

flat = tf.contrib.layers.flatten(conv)
square1 = flat*flat
pool = tf.layers.dense(square1, units = 100, name='dense1')
square2 = pool*pool
output = tf.layers.dense(square2, units = 10, name='dense2')
model = tf.sigmoid(output)
loss = tf.reduce_sum((y-model)*(y-model))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

#dense1_kernel_ = np.zeros(shape=(845,100))
#dense1_bias_ = np.zeros(shape=(100))
#dense2_kernel_ = np.zeros(shape=(100,10))
#dense2_bias_ = np.zeros(shape=(10))

t_moduli = np.array([40961, 65537, 114689, 147457, 188417])

with tf.Session() as sess:
    #acquisizione pesi
    saver.restore(sess, './nn_data/net-1')
    
    with tf.variable_scope('dense1', reuse=True):
        dense1_kernel_ = sess.run(tf.get_variable('kernel'))
        dense1_bias_ = sess.run(tf.get_variable('bias'))
    with tf.variable_scope('dense2', reuse=True):
        dense2_kernel_ = sess.run(tf.get_variable('kernel'))
        dense2_bias_ = sess.run(tf.get_variable('bias'))
    with tf.variable_scope('convolution', reuse=True):
        conv_kernel_ = sess.run(tf.get_variable('kernel'))
        conv_bias_ = sess.run(tf.get_variable('bias'))
    encoded_input_ = input_layer.eval({x: mnist.test.images})
    test_output = output.eval({x: mnist.test.images})

input_size = encoded_input_.shape[0]

encoded_input = np.empty(shape=(input_size,29,29,5), dtype=np.uint64)
dense1_kernel = np.empty(shape=(845,100,5), dtype=np.uint64)
dense1_bias = np.empty(shape=(100,5), dtype=np.uint64)
dense2_kernel = np.empty(shape=(100,10,5), dtype=np.uint64)
dense2_bias = np.empty(shape=(10,5), dtype=np.uint64)
conv_kernel = np.empty(shape=(5,5,5,5), dtype=np.uint64)
conv_bias = np.empty(shape=(5,5), dtype=np.uint64)

print("Weights processing...")
precisione = 100
for i in range(845):
    for j in range(100):
        temp = round(dense1_kernel_[i, j]*precisione)
        for t in range(5):
            dense1_kernel[i, j, t] = temp % t_moduli[t]
for i in range(100):
    for j in range(10):
        temp = round(dense2_kernel_[i, j]*precisione)
        for t in range(5):
            dense2_kernel[i, j, t] = temp % t_moduli[t]
for i in range(100):
    temp = round(dense1_bias_[i]*precisione)
    for t in range(5):
        dense1_bias[i, t] = temp % t_moduli[t]
for i in range(10):
    temp = round(dense2_bias_[i]*precisione)
    for t in range(5):
        dense2_bias[i, t] = temp % t_moduli[t]
for i in range(5):
    temp = round(conv_bias_[i]*precisione)
    for t in range(5):
        conv_bias[i, t] = temp % t_moduli[t]
for i in range(5):
    for j in range(5):
        for k in range(5):
            #conv_kernel_[i, j, 0, k] = round(conv_kernel_[i, j, 0, k]*precisione)
            temp = round(conv_kernel_[i, j, 0, k]*precisione)
            for t in range(5):
                conv_kernel[i, j, k, t] = temp % t_moduli[t]
print("Done.")


print("Input processing...")
for i in range(input_size):
    for j in range(29):
        for k in range(29):
            #encoded_input_[i,j,k,0] = round(encoded_input_[i,j,k,0]*precisione)
            temp = round(encoded_input_[i,j,k,0]*precisione)
            for t in range(5):
                encoded_input[i,j,k,t] = temp % t_moduli[t]
            
print("Done.")

np.save("./matrices/encoded_input", encoded_input)
np.save("./matrices/dense1_kernel", dense1_kernel)
np.save("./matrices/dense1_bias", dense1_bias)
np.save("./matrices/dense2_kernel", dense2_kernel)
np.save("./matrices/dense2_bias", dense2_bias)
np.save("./matrices/conv_kernel", conv_kernel)
np.save("./matrices/conv_bias", conv_bias)

print("Matrices saved")