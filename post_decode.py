
# Compare and decode plain and decrypted tensors. Compare predictions to floating point numbers
# in TensorFlow to check the loss of accuracy.

import os.path

plain_output_exists = os.path.isfile("./output_data/plain_layer_5.npy")
decrypted_output_exists = os.path.isfile("./output_data/decrypted_layer_5.npy")

if ((not plain_output_exists) and (not decrypted_output_exists)):
	print("Files missing, quitting.")
	exit()

import numpy as np
from wrapper import t_list
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Declaring the same Neural Network to restore the session and get the tensors
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

def extended_Euclidean_algorithm(a, b):
	b0 = b
	x0, x1 = 0, 1
	if b == 1: return 1
	while a > 1:
		q = a // b
		a, b = b, a%b
		x0, x1 = x1 - q * x0, x0
	if x1 < 0: x1 += b0
	return x1

def chinese_remainder_theorem(array):
	result = 0
	for t_index in range(len(array)):
		result += array[t_index].item() * bezout_coefficients[t_index] * t_product_over_t[t_index]
	return result % t_product

def crt_inverse(tensor):
	examples_count = tensor.shape[0]
	temp = np.empty(tensor.shape[:-1], dtype=object)
	for i in range(examples_count):
		for j in range(10):
			temp[i, j] = chinese_remainder_theorem(plain_output[i, j, :])
			if (temp[i, j]>negative_threshold):
				temp[i, j] = temp[i, j] - t_product
	return temp



# CRT PARAMETERS
# compute the producte of all t, and the threshold for negative numbers:
#   t_product
#   negative_threshold
t_product = 1
for t_index in range(len(t_list)):
	t_product = t_product * t_list[t_index]
negative_threshold = t_product // 2
# compute t_product // t and the Bezout coefficients, for all t: 
#   t_product_over_t
#   bezout_coefficients
t_product_over_t = []
bezout_coefficients = []
for t_index in range(len(t_list)):
	t_product_over_t.append(t_product // t_list[t_index])
	temp = extended_Euclidean_algorithm(t_product_over_t[t_index], t_list[t_index])
	bezout_coefficients.append(temp)



# COMPUTE PREDICTIONS
string = ""
if (plain_output_exists):
	plain_output = np.load("./output_data/plain_layer_5.npy")
if (decrypted_output_exists):
	decrypted_output = np.load("./output_data/decrypted_layer_5.npy")
	if (plain_output_exists):
		if (np.array_equal(plain_output, decrypted_output)):
			print("---- Plain and decrypted outputs coincide ----")
			cn_predictions = crt_inverse(decrypted_output)
			string = "Accuracy with SEAL encryption:"
		else:
			print("---- Plain and decrypted outputs are different. Computing accuracy with plain output ----")
			cn_predictions = crt_inverse(plain_output)
			string = "Accuracy with integer numbers (no encryption):"
	else:
		print("---- plain_layer_5.npy file is missing. Can't compare decrypted and plain outputs ----")
		cn_predictions = crt_inverse(decrypted_output)
		string = "Accuracy with SEAL encryption:"
else:
	print("---- decrypted_layer_5.npy file is missing. Can't compare decrypted and plain outputs ----")
	print("Computing accuracy with plain output...")
	cn_predictions = crt_inverse(plain_output)
	string = "Accuracy with integer numbers (no encryption):"
cn_predictions = np.argmax(cn_predictions, axis=1)



# PRINT ACCURACIES
with tf.Session() as sess:
	saver.restore(sess, './nn_data/net-50')

	tf_guessed_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	tf_accuracy = tf.reduce_mean(tf.cast(tf_guessed_predictions, "float"))
	cn_guessed_predictions = tf.equal(cn_predictions, tf.argmax(y, 1))
	cn_accuracy = tf.reduce_mean(tf.cast(cn_guessed_predictions, "float"))


	tf_guessed_predictions_ = tf_guessed_predictions.eval({x: mnist.test.images, y: mnist.test.labels})
	cn_guessed_predictions_ = cn_guessed_predictions.eval({y: mnist.test.labels})
	swapped_predictions_count = 0
	for i in range(tf_guessed_predictions_.size):
		if (tf_guessed_predictions_[i]!=cn_guessed_predictions_[i]):
			swapped_predictions_count = swapped_predictions_count + 1


	print("Accuracy with tensorflow and no CRT:", tf_accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
	print(string, cn_accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
	print("Number of swapped predictions: ", swapped_predictions_count)
