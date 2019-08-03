
# Pre-encoding of the Neural Networks weights and the input data. Pre-encoding means
# converting float numbers n to int((n * precision) % t_i), where t_i are the plain moduli.

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from wrapper import t_list

# Declare the same Neural Network to restore the session and get the tensors
learning_rate = 0.001
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
paddings = tf.constant([[0, 0], [1, 0,], [1, 0]])
input_layer = tf.reshape(x, [-1, 28, 28])
input_layer = tf.pad(input_layer, paddings, "CONSTANT")
input_layer = tf.reshape(input_layer, [-1, 29, 29, 1])
conv = tf.layers.conv2d(inputs=input_layer, filters=5, kernel_size=[5, 5], strides=[2, 2], padding="valid", activation=None, name='convolution')
flat = tf.contrib.layers.flatten(conv)
square1 = flat*flat
pool = tf.layers.dense(square1, units = 100, name='dense1')
square2 = pool*pool
output = tf.layers.dense(square2, units = 10, name='dense2')
model = tf.sigmoid(output)
loss = tf.reduce_sum((y-model)*(y-model))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Restore session
with tf.Session() as sess:
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

# Declare the outputs
input_size = encoded_input_.shape[0]
encoded_input = np.empty(shape=(input_size,29,29,5), dtype=np.uint64)
dense1_kernel = np.empty(shape=(845,100,5), dtype=np.uint64)
dense1_bias = np.empty(shape=(100,5), dtype=np.uint64)
dense2_kernel = np.empty(shape=(100,10,5), dtype=np.uint64)
dense2_bias = np.empty(shape=(10,5), dtype=np.uint64)
conv_kernel = np.empty(shape=(5,5,5,5), dtype=np.uint64)
conv_bias = np.empty(shape=(5,5), dtype=np.uint64)

# WEIGHTS
print("Weights processing...")
precision = 100
for i in range(845):
	for j in range(100):
		value = round(dense1_kernel_[i, j].item()*precision)
		for t in range(5):
			dense1_kernel[i, j, t] = value % t_list[t]
for i in range(100):
	for j in range(10):
		value = round(dense2_kernel_[i, j].item()*precision)
		for t in range(5):
			dense2_kernel[i, j, t] = value % t_list[t]
for i in range(100):
	value = round(dense1_bias_[i].item()*precision)
	for t in range(5):
		dense1_bias[i, t] = value % t_list[t]
for i in range(10):
	value = round(dense2_bias_[i].item()*precision)
	for t in range(5):
		dense2_bias[i, t] = value % t_list[t]
for i in range(5):
	value = round(conv_bias_[i].item()*precision)
	for t in range(5):
		conv_bias[i, t] = value % t_list[t]
for i in range(5):
	for j in range(5):
		for k in range(5):
			value = round(conv_kernel_[i, j, 0, k].item()*precision)
			for t in range(5):
				conv_kernel[i, j, k, t] = value % t_list[t]
print("Done.")

# INPUT
print("Input processing...")
for i in range(input_size):
	for j in range(29):
		for k in range(29):
			value = round(encoded_input_[i,j,k,0].item()*precision)
			for t in range(5):
				encoded_input[i,j,k,t] = value % t_list[t]
			
print("Done.")

# Store the encoded tensors
np.save("./output_data/plain_layer_0", encoded_input)
np.save("./nn_data/dense1_kernel", dense1_kernel)
np.save("./nn_data/dense1_bias", dense1_bias)
np.save("./nn_data/dense2_kernel", dense2_kernel)
np.save("./nn_data/dense2_bias", dense2_bias)
np.save("./nn_data/conv_kernel", conv_kernel)
np.save("./nn_data/conv_bias", conv_bias)

print("Tensors stored")