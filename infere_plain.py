
# Inference with NumPy and integers. Use of the Chinese Remainder Theorem to encode large numbers.
# Comparison with floating point numbers in TensorFlow to check the loss of accuracy.

import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from math import floor
import tensorflow as tf

t_list = [40961, 65537, 114689, 147457, 188417]
t_n = [208830439272397398017, 130520219464373862401, 74583470280817426433, 58009478173546659841, 45398788978896117761]
t_prod = 8553903623036669820174337 # > 2**80
t_mul_inv = [10665, 43884, 94414, 54859, 164822]

#t_mul_inv = []
#for t in range(5):
#    t_mul_inv.append(mul_inv(t_n[t], t_list[t]))

def chinese_remainder(array):
    risultato = 0
    for t in range(5):
        risultato += array[t].item() * t_mul_inv[t] * t_n[t]
    return risultato % t_prod

def mul_inv(a, b):
    b0 = b
    x0, x1 = 0, 1
    if b == 1: return 1
    while a > 1:
        q = a // b
        a, b = b, a%b
        x0, x1 = x1 - q * x0, x0
    if x1 < 0: x1 += b0
    return x1

# Set parameters
learning_rate = 0.001

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

paddings = tf.constant([[0, 0], [1, 0,], [1, 0]])
input_layer = tf.reshape(x, [-1, 28, 28])
input_layer = tf.pad(input_layer, paddings, "CONSTANT") 
input_layer = tf.reshape(input_layer, [-1, 29, 29, 1])

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
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

encoded_input = np.load("./output_data/encoded_input.npy")
dense1_kernel = np.load("./nn_data/dense1_kernel.npy")
dense1_bias = np.load("./nn_data/dense1_bias.npy")
dense2_kernel = np.load("./nn_data/dense2_kernel.npy")
dense2_bias = np.load("./nn_data/dense2_bias.npy")
conv_kernel = np.load("./nn_data/conv_kernel.npy")
conv_bias = np.load("./nn_data/conv_bias.npy")

with tf.Session() as sess:
    saver.restore(sess, './nn_data/net-1')
    input_size = encoded_input.shape[0]

    if False: #ELOTRO
        # LAYER 1: convolution + flatten
        print("Calculation of the convolution layer:")
        ris = np.empty((input_size,845,5), dtype=np.uint64)
        for i in range(input_size):
            if ((i%1000)==0):
                print(str(i/100)+"%")
            for j in range(5):
                for k in range(13):
                    for l in range(13):
                        for t in range(5):
                            temp_sum = 0
                            for x_index in range(5):
                                for y_index in range(5):
                                    temp_sum = temp_sum + encoded_input[i, k*2+x_index, l*2+y_index, t].item() * conv_kernel[x_index,y_index,j,t].item()
                                    temp_sum = temp_sum % t_list[t]
                            temp_sum = temp_sum + conv_bias[j, t].item()
                            temp_sum = temp_sum % t_list[t]
                            ris[i, j + (l*5) + (k*65), t] = temp_sum 
        print("100%")
        print("Calculation of the remaining layers")

        print("layer 1 type =", type(ris), type(ris[0,0,0]))
        np.save("./output_data/plain_layer_1", ris)

        
        # LAYER 2: square activation function
        ris = ris*ris
        for t in range(5):
            ris[...,t] = ris[...,t] % t_list[t]

        print("layer 2 type =", type(ris), type(ris[0,0,0]))
        np.save("./output_data/plain_layer_2", ris)

        # LAYER 3: fully connected layer
        #
        # Usually using the numpy.dot() function results in changing the dtype from uint64 to float64. So we should use dtype=object insted,
        # with the pythong integer object. But we always have small numbers: all of them are less then t_list[4] = 188417 < 4503599627370496 = 2**52.
        # So they can fit in the 52 bits mantissa of the float64 type. This allows us to NOT use dtype=objecy, which is slower.
        temp = np.empty((input_size,100,5), dtype=np.uint64)
        for t in range(5):
            temp[...,t] = ris[...,t].dot(dense1_kernel[...,t]) % t_list[t]
            temp[...,t] = temp[...,t] + dense1_bias[...,t] % t_list[t]
        ris = temp
        temp = None

        print("layer 3 type =", type(ris), type(ris[0,0,0]))
        np.save("./output_data/plain_layer_3", ris)

        # LAYER 4: square activation function
        ris = ris*ris
        for t in range(5):
            ris[...,t] = ris[...,t] % t_list[t]

        print("layer 4 type =", type(ris), type(ris[0,0,0]))
        np.save("./output_data/plain_layer_4", ris)

        # LAYER 5: fully connected layer
        temp = np.empty((input_size,10,5), dtype=np.uint64)
        for t in range(5):
            temp[...,t] = ris[...,t].dot(dense2_kernel[...,t]) % t_list[t]
            temp[...,t] = temp[...,t] + dense2_bias[...,t] % t_list[t]
        ris = temp
        temp = None

        print("layer 5 type =", type(ris), type(ris[0,0,0]))
        np.save("./output_data/plain_layer_5", ris)

        # CTR INVERSE
        negative_threshold = t_prod // 2
        temp = np.empty((input_size,10), dtype=object)
        for i in range(input_size):
            for j in range(10):
                temp[i, j] = chinese_remainder(ris[i, j, :])
                if (temp[i, j]>negative_threshold):
                    temp[i, j] = temp[i, j] - t_prod
        ris = temp
        temp = None

        # COMPUTE PREDICTIONS
        ris = np.argmax(ris, axis=1)

        print("final_predictions type =", type(ris), type(ris[0]))
        np.save("./output_data/final_predictions", ris)

    ris = np.load("./matrices/6_argmax.npy")

    # COMPUTE FINAL OUTPUT AND PRINT IT
    tf_predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    tf_predictions = tf_predictions.eval({x: mnist.test.images, y: mnist.test.labels})
    tf_accuracy = tf.reduce_mean(tf.cast(tf_predictions, "float"))
    crt_predictions = tf.equal(ris, tf.argmax(y, 1))
    crt_predictions = crt_predictions.eval({y: mnist.test.labels})
    crt_accuracy = tf.reduce_mean(tf.cast(crt_predictions, "float"))

    swapped_predictions_count = 0
    for i in range(tf_predictions.size):
        if (tf_predictions[i]!=crt_predictions[i]):
            swapped_predictions_count = swapped_predictions_count + 1

    print("Accuracy with tensorflow (no CRT):", tf_accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy with only numpy and CRT: ", crt_accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Number of swapped predictions: ", c)