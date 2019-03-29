# -*- coding: utf-8 -*-


import os

import tensorflow as tf
import numpy as np

from base.network import Network


class VGG(Network):
    def __init__(self, name_npy="vgg16.npy"):
        path_cur = os.path.dirname(__file__)
        path_par = os.path.split(path_cur)[0]
        path = os.path.join(path_par, "data", name_npy)
        self._data = np.load(path, encoding='latin1').item()

    def build(self):
        self.image = tf.placeholder(tf.float32, [1, None, None, 3])
        self._conv1_1 = self._conv2d(self.image, "conv1_1", False)
        self._conv1_2 = self._conv2d(self._conv1_1, "conv1_2", False)
        self._pool1 = self._max_pool(self._conv1_2, "pool1")

        self._conv2_1 = self._conv2d(self._pool1, "conv2_1", False)
        self._conv2_2 = self._conv2d(self._conv2_1, "conv2_2", False)
        self._pool2 = self._max_pool(self._conv2_2, "pool2")

        self._conv3_1 = self._conv2d(self._pool2, "conv3_1")
        self._conv3_2 = self._conv2d(self._conv3_1, "conv3_2")
        self._conv3_3 = self._conv2d(self._conv3_2, "conv3_3")
        self._pool3 = self._max_pool(self._conv3_3, "pool3")

        self._conv4_1 = self._conv2d(self._pool3, "conv4_1")
        self._conv4_2 = self._conv2d(self._conv4_1, "conv4_2")
        self._conv4_3 = self._conv2d(self._conv4_2, "conv4_3")
        self._pool4 = self._max_pool(self._conv4_3, "pool4")

        self._conv5_1 = self._conv2d(self._pool4, "conv5_1")
        self._conv5_2 = self._conv2d(self._conv5_1, "conv5_2")
        self._conv5_3 = self._conv2d(self._conv5_2, "conv5_3")

        self.feature = self._conv5_3  # [1, None, None, 512]

    def _conv2d(self, input, name, is_trainable=True):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            filter = tf.get_variable("filter", initializer=self._data[name][0], trainable=is_trainable)
            biases = tf.get_variable("biases", initializer=self._data[name][1], trainable=is_trainable)
            conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1],
                                padding='SAME')
            conv_bias = tf.nn.bias_add(conv, biases)
            conv_relu = tf.nn.relu(conv_bias)

        return conv_relu

    def _max_pool(self, input, name):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)
