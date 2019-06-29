import sys
#import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Set parameters
learning_rate = 0.001
training_epochs = 1#30
batch_size = 100
display_step = 2


x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes

#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#model = tf.nn.softmax(tf.matmul(x, W) + b)

paddings = tf.constant([[0, 0], [1, 0,], [1, 0]])
input_layer = tf.reshape(x, [-1, 28, 28])
input_layer = tf.pad(input_layer, paddings, "CONSTANT") 
input_layer = tf.reshape(input_layer, [-1, 29, 29, 1])
#TEST SUL PADDING
#test = [[0 for i in range(48)] for j in range(48)] 
#for i in range(48):
#    for j in range(48):
#        test[i][j] = input_layer[3][i][j][0]
#a = input_layer[3]
#sess = tf.Session()
#batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#im = np.array(sess.run(test, feed_dict={x: batch_xs, y: batch_ys}))
#im2 = np.array(sess.run(b, feed_dict={x: batch_xs, y: batch_ys}))
#print("Numero:", sess.run(tf.argmax(y, 1)[3],feed_dict={x: batch_xs, y: batch_ys}))
#sess.close()
#plt.imshow(im, cmap=plt.cm.Greys)
#plt.show()
#plt.imshow(im2, cmap=plt.cm.Greys)
#plt.show()



# Input Tensor Shape: [batch_size, 29, 29, 1]
# Output Tensor Shape: [batch_size, 13, 13, 5]
conv = tf.layers.conv2d(
    inputs=input_layer,
    filters=5,
    kernel_size=[5, 5],
    strides=[2, 2],
    padding="valid",
    activation=None,
    name='convolution') #tf.nn.relu

flat = tf.contrib.layers.flatten(conv)
square1 = flat*flat
pool = tf.layers.dense(square1, units = 100, name='dense1')
square2 = pool*pool
output = tf.layers.dense(square2, units = 10, name='dense2')
model = tf.sigmoid(output) #tf.nn.softmax()
#loss = -tf.reduce_sum(y*tf.log(model))
loss = tf.reduce_sum((y-model)*(y-model))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        number_of_iterations = int(mnist.train.num_examples/batch_size)
        for i in range(number_of_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})/number_of_iterations
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    saved_path = saver.save(sess, './nn_data/net', global_step=training_epochs)

    print("Training completed!")

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    predictions2 = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy2 = tf.reduce_mean(tf.cast(predictions2, "float"))
    abla1 = predictions.eval({x: mnist.test.images, y: mnist.test.labels})
    abla2 = predictions2.eval({x: mnist.test.images, y: mnist.test.labels})
    c = 0
    for i in range(len(abla1)):
        if (abla1[i]!=abla2[i]):
            c = c + 1
    print(c)
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy without sigmoide:", accuracy2.eval({x: mnist.test.images, y: mnist.test.labels}))
