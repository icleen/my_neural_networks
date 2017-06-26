from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import prep_hiragana
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# making the onehot labels for the hiragana data
def onehot_labels(list):
    out = np.zeros(shape=(len(list), 45), dtype=np.int32)
    for i, item in enumerate(list):
        out[i][int(item)] = 1
    return out

# setting up the cnn
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main(_):
    # Import data
    prepper = prep_hiragana.prepper('hiragana', 'hiragana.txt')
    training = prepper.train_images()
    t_labels = np.asarray(prepper.train_labels(), dtype=np.int32)
    validation = prepper.validate_images()
    v_labels =  np.asarray(prepper.validate_labels(), dtype=np.int32)
    v_labels = onehot_labels(v_labels)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 1024])
    W = tf.Variable(tf.zeros([1024, 45]))
    b = tf.Variable(tf.zeros([45]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 45])

    # adding in the convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,32,32,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # adding the second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #adding the final layer
    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # adding the dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # adding the readout layer
    W_fc2 = weight_variable([1024, 45])
    b_fc2 = bias_variable([45])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    for i in range(3000):
        a = i*50 % len(training)
        batchx = training[a:a + 50]
        batchy = t_labels[a:a + 50]
        batchy = onehot_labels(batchy)
        train_step.run(feed_dict={x: batchx, y_: batchy, keep_prob: 0.5})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batchx, y_: batchy, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: validation, y_: v_labels, keep_prob: 1.0}))

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: validation, y_: v_labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
