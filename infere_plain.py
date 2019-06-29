
# INFERENZA con numpy e numeri interi
# CON CTR
# CONTRONTA CON CTR E SENZA CTR

import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from math import floor
import tensorflow as tf


from functools import reduce
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


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set parameters
learning_rate = 0.001
training_epochs = 1#30
batch_size = 100
display_step = 2


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

encoded_input = np.load("./matrices/encoded_input.npy")
dense1_kernel = np.load("./matrices/dense1_kernel.npy")
dense1_bias = np.load("./matrices/dense1_bias.npy")
dense2_kernel = np.load("./matrices/dense2_kernel.npy")
dense2_bias = np.load("./matrices/dense2_bias.npy")
conv_kernel = np.load("./matrices/conv_kernel.npy")
conv_bias = np.load("./matrices/conv_bias.npy")
t_moduli = [40961, 65537, 114689, 147457, 188417]
t_n = [208830439272397398017, 130520219464373862401, 74583470280817426433, 58009478173546659841, 45398788978896117761]
t_prod = 8553903623036669820174337
t_mul_inv = []
for t in range(5):
    t_mul_inv.append(mul_inv(t_n[t], t_moduli[t]))

with tf.Session() as sess:
    saver.restore(sess, './nn_data/net-1')
    input_size = encoded_input.shape[0]

    #convoluzione + flatten
    if False: #ELOTRO
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
                                    temp_sum = temp_sum % t_moduli[t]
                            temp_sum = temp_sum + conv_bias[j, t].item()
                            temp_sum = temp_sum % t_moduli[t]
                            ris[i, j + (l*5) + (k*65), t] = temp_sum 

        print("100%")
        print("Calculation of the remaining layers")

        #funzione attivazione: al quadrato
        ris = ris*ris
        for t in range(5):
            ris[...,t] = ris[...,t] % t_moduli[t]

        #ris = np.empty((input_size,845,5), dtype=np.uint64)

        #fully connected layer 1
        temp = np.empty((input_size,100,5), dtype=np.uint64)
        for t in range(5):
            temp[...,t] = ris[...,t].dot(dense1_kernel[...,t]) % t_moduli[t]
            temp[...,t] = temp[...,t] + dense1_bias[...,t] % t_moduli[t]
        ris = temp
        temp = None

        #funzione attivazione: al quadrato
        ris = ris*ris
        for t in range(5):
            ris[...,t] = ris[...,t] % t_moduli[t]

        #fully connected layer 2
        temp = np.empty((input_size,10,5), dtype=np.uint64)
        for t in range(5):
            temp[...,t] = ris[...,t].dot(dense2_kernel[...,t]) % t_moduli[t]
            temp[...,t] = temp[...,t] + dense2_bias[...,t] % t_moduli[t]
        ris = temp
        temp = None

        #tests
        ris = np.load("./matrices/5_0_dense2_kernel.npy")

        for t in range(5):
            ris[...,t] = ris[...,t] + dense2_bias[...,t] % t_moduli[t]

    ris = np.load("./matrices/5_1_dense2_bias.npy")

    #temp = (temp==ris)


#    for i in range(10000):
#        for j in range(10):
#            for k in range(5):
#                if (not temp[i,j,k]):
#                    print("falso")
#                    exit()

    print("Fine")
    exit()
    print("waiting...")

    #fine tests
    #ris = np.load("./matrices/5_1_dense2_bias.npy")

    #CTR INVERSE
    negative_threshold = t_prod // 2
    temp = np.empty((input_size,10), dtype=object)
    for i in range(input_size):
        for j in range(10):
            temp[i, j] = chinese_remainder(ris[i, j, :])
            if (temp[i, j]>negative_threshold):
                temp[i, j] = temp[i, j] - t_prod
    ris = temp
    temp = None

    np.save("./matrices/5_2_decoded.npy", ris)

    #trasforma in predizione
    ris = np.argmax(ris, axis=1)
    np.save("./matrices/6_argmax.npy", ris)


    #test sugli output e stampa
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    predictions2 = tf.equal(ris, tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(predictions2, "float"))
    abla1 = predictions.eval({x: mnist.test.images, y: mnist.test.labels})
    abla2 = predictions2.eval({y: mnist.test.labels})

    c = 0
    for i in range(len(abla1)):
        if (abla1[i]!=abla2[i]):
            c = c + 1

    print("Numero esatto di predizioni invertite: ", c)
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy without tensorflow:", accuracy2.eval({x: mnist.test.images, y: mnist.test.labels}))