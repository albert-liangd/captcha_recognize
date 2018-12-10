#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-04 16:15
# @Author  : Albert liang
# @Email   : ld602199512@gmail.com
# @File    : model.py

import tensorflow as tf


class CNN(object):
    def __init__(self, image_height, image_width, kernels, drop_keep_prob):
        """

        :param image_height: the size of image height [type: int]
        :param image_width: the size of image width [type: int]
        :param kernels: the kernels list for convolution filters [type:list] like [64,64,128]
        :param drop_keep_prob:
        """
        self.height = image_height
        self.width = image_width
        self.kernels = kernels
        self.drop = drop_keep_prob

    def _conv2d(self, value, weight):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')

    def _weight_init(self, name, shape):
        """weight_variable generates a weight variable of a given shape."""
        with tf.device('/cpu:0'):
            initializer = tf.truncated_normal_initializer(stddev=0.1)
            return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

    def _bias_init(self, name, shape):
        with tf.device("/cpu:0"):
            initializer = tf.constant_initializer(0.1)
            return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

    def _max_pool_2x2(self, value, name):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def inference(self, images, keep_prob):

        # reshape shape
        images = tf.reshape(images, [-1, self.height, self.width, 1])

        # add  first convolution layer
        with tf.variable_scope("conv1")as scope:
            kernel = self._weight_init("weights", shape=[3, 3, 1, self.kernels[0]])
            biases = self._bias_init("baises", [self.kernels[0]])
            pre_activated = tf.nn.bias_add(self._conv2d(images, kernel), biases)
            conv1 = tf.nn.relu(pre_activated, name=scope.name)

        # add the first pooling layer
        pool1 = self._max_pool_2x2(conv1, name="pool1")

        # add  second convolution layer
        with tf.variable_scope("conv2") as scope:
            kernel = self._weight_init("weights", shape=[3, 3, self.kernels[0], self.kernels[1]])
            biases = self._bias_init("baises", [self.kernels[1]])
            pre_activated = tf.nn.bias_add(self._conv2d(pool1, kernel), biases)
            conv2 = tf.nn.relu(pre_activated, name=scope.name)

        # add the second pooling layer
        pool2 = self._max_pool_2x2(conv2, name="pool2")

        # add  second convolution layer
        with tf.variable_scope("conv2") as scope:
            kernel = self._weight_init("weights", shape=[3, 3, self.kernels[1], self.kernels[2]])
            biases = self._bias_init("baises", [self.kernels[2]])
            pre_activated = tf.nn.bias_add(self._conv2d(pool2, kernel), biases)
            conv3 = tf.nn.relu(pre_activated, name=scope.name)

        # add the second pooling layer
        pool3 = self._max_pool_2x2(conv3, name="pool3")

        with tf.variable_scope("local1") as scope:
            batch_size = images.get_shape()[0].value
            reshape = tf.reshape(pool3, [batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self._weight_init('weights', shape=[dim, 1024])
            biases = self._bias_init("biases", [1024])
            local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # add dropout
        local1_drop = tf.nn.dropout(local1, self.drop)









    def _loss(self, logits, labels):
        pass

    def test(self):
        pass
