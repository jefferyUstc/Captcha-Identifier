#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2018/11/25 18:53'
import os
import re
import time
import numpy as np
import cv2

base_path = r'D:\java-oxygen\code\outsource\05'  # windows path
# base_path = ''  # linux path
folder = 'images'
characters = 'abcdefghijklmnopqrstuvwxyz'


def load_file_list(path=None, regx='\.png', printable=True, label='[a-zA-Z]+'):
    """Return a file list in a folder by given a path and regular expression.

    Parameters
    ----------
    path : a string or None
        A folder path.
    regx : a string
        The regx of file name.
    label : s string
        The regx of label.
    printable : boolean, whether to print the files infomation.
    """
    if not path:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
                return_list.append((os.path.join(path, f), re.search(label, f).group()))
    if printable:
        print('Match file example = %s' % (str(return_list[0])))
        print('Number of files = %d in path: %s' % (len(return_list), path))
    return np.array(return_list)


def _process_batch_label(batch_labels):
    """
    process labels

    :param batch_labels: labels
    :return: normalized labels
    """
    batch_num = len(batch_labels)
    final_labels = np.zeros([batch_num, 4, 26])  # four character each captcha
    for i in range(batch_num):
        for j, ch in enumerate(batch_labels[i]):
            final_labels[i, j, characters.find(ch.lower())] = 1
    final_labels = np.reshape(final_labels, [batch_num, 4*26])
    return final_labels


def _process_batch_img(batch_files, resize_shape=None):
    """
    read img file
    :param batch_files:
    :param resize_shape:
    :return: normalized batch imgs
    """

    batch_data = []
    for path in batch_files:
        img = cv2.imread(path, flags=0)
        if resize_shape:
            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)
        # task specific
        img[img < 193] = 0
        img[img >= 193] = 1
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])  # no_resize (100, 120)
        batch_data.append(img)
    return np.array(batch_data)


def get_batch_data(batch_size=50, shuffle=True, epoch=5, resize_shape=None):
    """
    ignore the last batch residual
    :param resize_shape: list, resize img to a certain shape
    :param batch_size:batch_size
    :param shuffle: whether shuffle per epoch
    :param epoch: epoch number
    :return: batch img generater
    """

    X_data = load_file_list(os.path.join(base_path, folder), printable=True)
    batch_per_epoch = len(X_data) // batch_size
    print('in this training, we train %d epoch, %d batch per epoch' %
          (epoch, batch_per_epoch))

    for i in range(epoch):
        if shuffle:
            np.random.shuffle(X_data)
        files, labels = X_data[:, 0], X_data[:, 1]
        for j in range(batch_per_epoch):
            yield _process_batch_img(files[j * batch_size:j * batch_size + batch_size], resize_shape), \
                  _process_batch_label(labels[j * batch_size:j * batch_size + batch_size])


if __name__ == '__main__':
    # unit test
    start = time.time()
    g = get_batch_data()  # <class 'generator'>
    data, labels = next(g)
    print(data.shape, labels.shape)  # (50, 100, 120, 1) (50, 104)
    print('finished in %d ms' % ((time.time() - start) * 1000.))

