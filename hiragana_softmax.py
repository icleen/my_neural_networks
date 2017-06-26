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

def onehot_labels(list):
    out = np.zeros(shape=(len(list), 45), dtype=np.int32)
    for i, item in enumerate(list):
        out[i][int(item)] = 1
    return out


def main(_):
    # Import data
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    prepper = prep_hiragana.prepper('hiragana', 'hiragana.txt')
    training = prepper.train_images()
    t_labels = np.asarray(prepper.train_labels(), dtype=np.int32)
    validation = prepper.validate_images()
    v_labels =  np.asarray(prepper.validate_labels(), dtype=np.int32)
    v_labels = onehot_labels(v_labels)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 900])
    W = tf.Variable(tf.zeros([900, 45]))
    b = tf.Variable(tf.zeros([45]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 45])

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
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    for i in range(1000):
        a = i*50 % len(training)
        batchx = training[a:a + 50]
        batchy = t_labels[a:a + 50]
        batchy = onehot_labels(batchy)
        sess.run(train_step, feed_dict={x: batchx, y_: batchy})
        if i%100 == 0:
            print(sess.run(accuracy, feed_dict={x: validation,
                                              y_: v_labels}))


    print(sess.run(accuracy, feed_dict={x: validation,
                                      y_: v_labels}))

if __name__ == '__main__':
    tf.app.run()
