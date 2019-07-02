import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

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
init = tf.global_variables_initializer()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Accuracy without sigmoide:", accuracy2.eval({x: mnist.test.images, y: mnist.test.labels}))
