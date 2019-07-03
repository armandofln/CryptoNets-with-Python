
# Infere whithin the encrypted polynomials space.

import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from math import floor
import tensorflow as tf

#from functools import reduce #probabilmente puo' essere cancellato


from wrapper import SEAL
import numpy as np

SEALobj = SEAL()
q_list = SEALobj.q_list
k_list = SEALobj.k_list
t_list = [40961, 65537, 114689, 147457, 188417]
t_n = [208830439272397398017, 130520219464373862401, 74583470280817426433, 58009478173546659841, 45398788978896117761]
t_prod = 8553903623036669820174337 # > 2**80
t_mul_inv = [10665, 43884, 94414, 54859, 164822]

#t_mul_inv = []
#for t in range(5):
#    t_mul_inv.append(mul_inv(t_n[t], t_list[t]))

def oggetto(tensoreuint):
    shape = tensoreuint.shape
    flatten(tensoreuint)
    new = np.empty((tensoreuint.size,), dtype=object)
    for i in range(tensoreuint.size):
        new[i] = int(tensoreuint[i].item())
    new.shape = shape
    return new

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


with tf.Session() as sess:
    saver.restore(sess, './nn_data/net-1')
    input_size = encoded_input.shape[0]

    if False: #ELOTRO
        print("Calculation of the convolution layer:")

        #convoluzione + flatten CRIPTATI
        encoded_input = cripta(encoded_input)
        poly_groups_count = 4097*4*2
        poly_groups_count = encoded_input.shape[0]//poly_groups_count

        input_size = encoded_input.shape[0]
        ris = np.empty((input_size,845,5), dtype=np.uint64)

        for poly_group_index in range(poly_groups_count):
            for s_index in range(2):
                for q_index in range(len(q_list)):
                    temp_sum = q_index + (s_index*4) + (poly_group_index*2*4)
                    temp_sum = temp_sum / (poly_groups_count*2*4)
                    temp_sum = round(temp_sum*1000)
                    print(str(temp_sum/10)+"%")
                    for n_index in range(4097): #elotro
                        index = n_index + (q_index*4097) + (s_index*4*4097) + (poly_group_index*2*4*4097)
                        for j in range(5):
                            for k in range(13):
                                for l in range(13):
                                    for t in range(5):
                                        temp_sum = 0
                                        for x_index in range(5):
                                            for y_index in range(5):
                                                temp_sum = temp_sum + encoded_input[index, k*2+x_index, l*2+y_index, t].item() * conv_kernel[x_index,y_index,j,t].item()
                                        if (n_index==0):
                                            if (s_index==0):
                                                temp_sum = temp_sum + ((conv_bias[j, t].item())*k_list[t][q_index])
                                        temp_sum = temp_sum % q_list[q_index]
                                        ris[index, j + (l*5) + (k*65), t] = temp_sum

        ris = decripta(ris)

        temp = np.load("./matrices/1_conv.npy")
        confronta(ris, temp)

        print("100%")
        print("Calculation of the remaining layers")

        #funzione attivazione1: al quadrato
        ris = ris*ris
        for t in range(5):
            ris[...,t] = ris[...,t] % t_list[t]

        #fully connected layer 1
        temp = np.empty((input_size,100,5), dtype=np.uint64)
        for t in range(5):
            temp[...,t] = ris[...,t].dot(dense1_kernel[...,t]) % t_list[t]
            temp[...,t] = temp[...,t] + dense1_bias[...,t] % t_list[t]
        ris = temp
        temp = None

        # square layer 2 CRIPTATO
        ris = np.load("./matrices/3_1_dense1_bias.npy")
        ris = SEALobj.encrypt_tensor(ris)
        ris = SEALobj.square_tensor(ris)
        ris = SEALobj.decrypt_tensor(ris, 10000)
        temp = np.load("./matrices/4_attivazione2.npy")
        SEALobj.check(ris, temp)

        # fully connected layer 2 CRIPTATO
        ris = np.load("./matrices/4_attivazione2.npy")
        ris = cripta(ris)
        ris = oggetto(ris)
        dense2_kernel = oggetto(dense2_kernel)
        dense2_bias = oggetto(dense2_bias)

        ## kernel
        temp = np.empty((ris.shape[0],10,5), dtype=object)
        for t in range(5):
            temp[...,t] = ris[...,t].dot(dense2_kernel[...,t])
        ris = temp
        temp = None

        ## % q
        poly_groups_count = 4097*4*2
        poly_groups_count = ris.shape[0]//poly_groups_count

        for axis1 in range(ris.shape[1]):
            for axis2 in range(ris.shape[2]):
                for gdp in range(poly_groups_count):
                    for size_index in range(2):
                        for q_index in range(4):
                            for n_index in range(4097):
                                axis0 = gdp*2*4*4097 + size_index*4*4097 + q_index*4097 + n_index
                                temp = ris[axis0,axis1,axis2]
                                temp = temp % q_list[q_index]
                                ris[axis0,axis1,axis2] = temp

        ## bias
        for axis1 in range(ris.shape[1]):
            for axis2 in range(ris.shape[2]):
                for gdp in range(poly_groups_count):
                    for q_index in range(4):
                        axis0 = gdp*4097*4*2 + (4097*q_index)
                        temp = ris[axis0,axis1,axis2]
                        temp = temp + dense2_bias[axis1,axis2]*k_list[axis2][q_index]
                        temp = temp % q_list[q_index]
                        ris[axis0,axis1,axis2] = temp

        ris = decripta(ris, 10000)

        temp = np.load("./matrices/5_1_dense2_bias.npy")
        confronta(ris, temp)

    #FINE TESTS
    SEALobj.deallocate()
    exit()
    print("waiting...")

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