#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2018/11/26 15:01'

import tensorflow as tf
import numpy as np
import cv2
import os
import mymodel
import time

width = 120
height = 100
char_num = 4
characters = 'abcdefghijklmnopqrstuvwxyz'
classes = 26


def predict_image(captcha):
    """
    predict captcha of single image

    :captcha: captcha img path or captcha img ndarray data
    :return: a string containing a captcha
    """

    if isinstance(captcha, np.ndarray):
        img = captcha
    elif isinstance(captcha, str):
        if os.path.exists(captcha):
            img = cv2.imread(captcha, 0)  # (100, 120)
        else:
            raise FileNotFoundError('captcha not exists')
    else:
        raise ValueError('the captcha param should be '
                         'a path of img or ndarray img data')

    img[img < 193] = 0
    img[img >= 193] = 1
    img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
    x = tf.placeholder(tf.float32, [None, height, width, 1])
    y_conv, _ = mymodel.captcha_model(x, keep_prob=1, trainable=False)
    predict = tf.argmax(tf.reshape(y_conv, [-1, char_num, classes]), 2)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    s = ''
    sess = tf.Session()
	# uncomment codes below if need to validate mutilple imgs in the same program
    # tf.reset_default_graph()
    sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state('./model_data')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('oh, bad model, please check')
    pre_list = sess.run(predict, feed_dict={x: img})

    for character in pre_list[0]:
        s += characters[character]
    return s


if __name__ == '__main__':
    start = time.time()
    # infer 45-55ms
    # load model 195-210ms
    s = predict_image('1.png')
    print(s)
    print('finished in %d ms' % ((time.time() - start) * 1000.))
