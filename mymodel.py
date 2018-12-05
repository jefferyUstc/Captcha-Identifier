#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2018/11/28 19:56'

import tensorflow as tf


# initializer = tf.truncated_normal_initializer(stddev=0.01)
initializer = tf.initializers.variance_scaling()


def conv2d(input_x, filters, kernel_size=(5, 5), activation=tf.nn.relu, strides=(1, 1),
           padding='SAME', trainable=True, name=None):
        conv_layer = tf.layers.conv2d(input_x, filters, kernel_size,
                                      strides, padding, activation=activation,
                                      kernel_initializer=initializer,
                                      bias_initializer=initializer,
                                      trainable=trainable, name=name)
        return conv_layer


def maxpool2d(input_x, padding='valid', pool_size=(2, 2), stride=(2, 2), name=None):
        return tf.layers.max_pooling2d(input_x, pool_size, stride, padding, name=name)


def dense(input_x, units, activation=tf.nn.relu,
          trainable=True, name=None):
        return tf.layers.dense(input_x, units,
                               activation,
                               kernel_initializer=initializer,
                               bias_initializer=initializer,
                               trainable=trainable,
                               name=name)


def captcha_model(x_img, keep_prob=0.75, trainable=True):
    """
    build model
    input (50, 100, 120, 1) (50, 104)
    :param trainable:
    :param x_img: input tensor
    :param keep_prob: dropout params
    :return: softmax_logits， logits
    """

    conv1 = conv2d(x_img, 8, trainable=trainable, name='conv1')
    pool1 = maxpool2d(conv1, name='pool1')

    conv2 = conv2d(pool1, 16, trainable=trainable, name='conv2')
    pool2 = maxpool2d(conv2, name='pool2')

    conv3 = conv2d(pool2, 32, trainable=trainable, name='conv3')
    pool3 = maxpool2d(conv3, name='pool3')

    conv4 = conv2d(pool3, 64, trainable=trainable, name='conv4')
    pool4 = maxpool2d(conv4, name='pool4')
    # print(pool4.get_shape())  # (50, 6, 7, 64)

    # flatten = tf.layers.flatten(pool4, name='flatten')
    flatten = tf.reshape(pool4,
                         [-1, pool4.get_shape()[1]*pool4.get_shape()[2]*pool4.get_shape()[3]],
                         name='flatten')
    # print(flatten.get_shape())  # (50, 2688)

    fc1 = dense(flatten, 1024, trainable=trainable, name='fc1')
    fc1 = tf.layers.dropout(fc1, rate=keep_prob, training=trainable, name='fc1-dropout')

    logits = dense(fc1, 104, activation=None, trainable=trainable, name='fc2')

    fc2 = tf.nn.softmax(logits)
    return fc2, logits


if __name__ == '__main__':
    # unit test
    x_input = tf.placeholder(dtype=tf.float32, shape=[50, 100, 120, 1], name='x_input')
    res = captcha_model(x_input)
    print(res[0].get_shape())  # (50, 104)







