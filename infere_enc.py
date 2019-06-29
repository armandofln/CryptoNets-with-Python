
# INFERENZA con numpy e numeri interi
# CON CTR
# CONTRONTA CON CTR E SENZA CTR

import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from math import floor
import tensorflow as tf

from functools import reduce


import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./SEAL/libseal.so')
lib.initialize()
lib.square.restype = ctypes.POINTER(ctypes.c_uint64)
lib.encrypt_tensor.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.decrypt_tensor.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.k_list.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
q_list = [36028797014376449, 36028797013327873, 1152921504241942529, 1152921504369344513]

#k_list = [3578163283476560, 5902913912171171, 256727058426082317, 114641959114678357]
#per t=40961
#k_list = [3578163283476560, 5902913912171171, 256727058426082317, 114641959114678357]
#per t=188417
#k_list = [23809558905221309, 3180918061091669, 313592054702576267, 922602767607469907]

temp = (ctypes.c_uint64 * 20)()
lib.k_list(temp)
k_list = []
for t in range(5):
    temp2 = []
    for k in range(4):
        temp2.append(int(temp[(t*4)+k]))
    k_list.append(temp2)
temp = None
temp2 = None

def flatten(tensore):
    dimensione = tensore.size
    tensore.shape = (dimensione,)


def confronta(tensoreA, tensoreB):
    print("")
    if (tensoreA.size!=tensoreB.size):
        print("I due tensori hanno dimensioni diverse:")
        print(tensoreA.shape)
        print(tensoreB.shape)
        return
    flatten(tensoreA)
    flatten(tensoreB)
    confronto = (tensoreA==tensoreB)
    for i in range(tensoreB.size):
        if (not confronto[i]):
            print("Differenza in i =", i)
            print(tensoreA[i])
            print(tensoreB[i])
            return
    print("Sono uguali!")

def oggetto(tensoreuint):
    shape = tensoreuint.shape
    flatten(tensoreuint)
    new = np.empty((tensoreuint.size,), dtype=object)
    for i in range(tensoreuint.size):
        new[i] = int(tensoreuint[i].item())
    new.shape = shape
    return new

def cripta(input_array):
    grado = 4096
    shape = input_array.shape
    shape_size = len(shape)
    q = 4
    assert shape[-1] == 5, "Errore in cripta: input_array deve avere size[-1] = 5"

    #calcolo dimensioni
    input_size = 1
    output_size = 1
    input_axis0_size = shape[0]
    output_axis0_size = 1
    data_size = 1

    for i in range(shape_size):
        input_size = input_size * shape[i]
    data_size = input_size // input_axis0_size
    divisione = input_axis0_size//grado
    resto = input_axis0_size%grado
    output_axis0_size = divisione * ((grado+1)*2*q)
    if (resto!=0):
        output_axis0_size = output_axis0_size + ((grado+1)*2*q)
    output_size = output_axis0_size * data_size

    #chiamata funzione
    input_array.shape = (input_size,)
    input_vec = (ctypes.c_uint64 * (input_size))()
    for i in range(input_size):
        input_vec[i] = input_array[i]
    output_vec = (ctypes.c_uint64 * (output_size))()
    data_size = data_size // 5 #adegua data size allo standard del wrap in c++
    lib.encrypt_tensor(input_vec, output_vec, input_axis0_size, output_axis0_size, data_size)

    #costruzione risultato
    output_array = np.empty((output_size), dtype=np.uint64)
    for i in range(output_size):
        output_array[i] = output_vec[i]
    shape = (output_axis0_size,) + shape[1:] #output shape
    output_array.shape = shape
    input_vec = None
    output_vec = None
    input_array = None
    return output_array

def decripta(input_array, output_axis0_size=10000):
    grado = 4096
    shape = input_array.shape
    shape_size = len(shape)
    q = 4
    assert (shape[0]%((grado+1)*4*2)) == 0, "Errore in decripta: input_array ha size[0] inaspettato"
    assert shape[-1] == 5, "Errore in decripta: input_array deve avere size[-1] = 5"

    #calcolo dimensioni
    input_size = 1
    output_size = 1
    input_axis0_size = shape[0]
    output_axis0_size = output_axis0_size
    data_size = 1

    for i in range(shape_size):
        input_size = input_size * shape[i]
    data_size = input_size // input_axis0_size
    output_size = output_axis0_size * data_size

    #chiamata funzione
    input_array.shape = (input_size,)
    input_vec = (ctypes.c_uint64 * (input_size))()
    for i in range(input_size):
        input_vec[i] = input_array[i]
    output_vec = (ctypes.c_uint64 * (output_size))()
    data_size = data_size // 5 #adegua data size allo standard del wrap in c++
    lib.decrypt_tensor(input_vec, output_vec, input_axis0_size, output_axis0_size, data_size)

    #costruzione risultato
    output_array = np.empty((output_size), dtype=np.uint64)
    for i in range(output_size):
        output_array[i] = output_vec[i]
    shape = (output_axis0_size,) + shape[1:] #output shape
    output_array.shape = shape
    input_vec = None
    output_vec = None
    input_array = None
    return output_array

def alquadrato(input_array):
    grado = 4096
    shape = input_array.shape
    shape_size = len(shape)
    q = 4
    assert (shape[0]%((grado+1)*4*2)) == 0, "Errore in decripta: input_array ha size[0] inaspettato"
    assert shape[-1] == 5, "Errore in decripta: input_array deve avere size[-1] = 5"

    #calcolo dimensioni
    input_size = 1
    input_axis0_size = shape[0]
    data_size = 1

    for i in range(shape_size):
        input_size = input_size * shape[i]
    data_size = input_size // input_axis0_size
    output_size = input_size
    output_axis0_size = input_axis0_size

    #chiamata funzione
    input_array.shape = (input_size,)
    input_vec = (ctypes.c_uint64 * (input_size))()
    for i in range(input_size):
        input_vec[i] = input_array[i]
    output_vec = (ctypes.c_uint64 * (output_size))()
    data_size = data_size // 5 #adegua data size allo standard del wrap in c++
    lib.square_tensor(input_vec, output_vec, input_axis0_size, output_axis0_size, data_size)

    #costruzione risultato
    output_array = np.empty((output_size), dtype=np.uint64)
    for i in range(output_size):
        output_array[i] = output_vec[i]
    #shape = (output_axis0_size,) + shape[1:] #output shape
    output_array.shape = shape
    input_vec = None
    output_vec = None
    input_array = None
    return output_array

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

        #funzione attivazione1: al quadrato
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

        # square layer 2 CRIPTATO
        ris = np.load("./matrices/3_1_dense1_bias.npy")
        ris = cripta(ris)
        ris = oggetto(ris)
        ris = alquadrato(ris)
        ris = decripta(ris, 10000)

        temp = np.load("./matrices/4_attivazione2.npy")
        confronta(ris, temp)

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
        polys_groups_count = 4097*4*2
        polys_groups_count = ris.shape[0]//polys_groups_count

        for axis1 in range(ris.shape[1]):
            for axis2 in range(ris.shape[2]):
                for gdp in range(polys_groups_count):
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
                for gdp in range(polys_groups_count):
                    for q_index in range(4):
                        axis0 = gdp*4097*4*2 + (4097*q_index)
                        temp = ris[axis0,axis1,axis2]
                        temp = temp + dense2_bias[axis1,axis2]*k_list[axis2][q_index]
                        temp = temp % q_list[q_index]
                        ris[axis0,axis1,axis2] = temp

        ris = decripta(ris, 10000)

        temp = np.load("./matrices/5_1_dense2_bias.npy")
        confronta(ris, temp)

    #convoluzione + flatten CRIPTATI TEST
    polys_groups_count = 4097*4*2
    polys_groups_count = 250*4*2#elotro
    polys_groups_count = ris.shape[0]//polys_groups_count

    input_size = encoded_input.shape[0]
    ris = np.empty((input_size,845,5), dtype=np.uint64)

    for poly_group_index in range(polys_groups_count):
        for s_index in range(2):
            for q_index in range(len(q_list)):
                temp_sum = q_index + (s_index*4) + (poly_group_index*2*4)
                temp_sum = temp_sum / (polys_groups_count*2*4)
                print(str(temp_sum*100)+"%")
                for n_index in range(250): #elotro
                    index = n_index + (q_index*250) + (s_index*4*250) + (poly_group_index*2*4*250)
                    for j in range(5):
                        for k in range(13):
                            for l in range(13):
                                for t in range(5):
                                    temp_sum = 0
                                    for x_index in range(5):
                                        for y_index in range(5):
                                            temp_sum = temp_sum + encoded_input[index, k*2+x_index, l*2+y_index, t].item() * conv_kernel[x_index,y_index,j,t].item()
                                            temp_sum = temp_sum % t_moduli[t]
                                    temp_sum = temp_sum + conv_bias[j, t].item()
                                    temp_sum = temp_sum % t_moduli[t]
                                    ris[index, j + (l*5) + (k*65), t] = temp_sum

    temp = np.load("./matrices/1_conv.npy")
    confronta(ris, temp)

    #FINE TESTS
    lib.deallocate()
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