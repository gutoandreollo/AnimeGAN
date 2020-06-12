import tensorflow as tf

import numpy as np
import sys

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self, vgg19_npy_path='../data/vgg19.npy'):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        else:
            self.data_dict = None
            print("npy file load error!")
            sys.exit(1)

    def build(self, rgb, include_fc=False):
        """
        load variable from npy to build the VGG
        input format: bgr image with shape [batch_size, h, w, 3]
        scale: (-1, 1)
        """

        rgb_scaled = ((rgb + 1) / 2) * 255.0 # [-1, 1] ~ [0, 255]

        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]])

        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self._conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self._max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")

        self.conv4_4_no_activation = self._no_activation_conv_layer(self.conv4_3, "conv4_4")

        self.conv4_4 = self._conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self._max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self._conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self._max_pool(self.conv5_4, 'pool5')

        if include_fc:
            self.fc6 = self._fc_layer(self.pool5, "fc6")
            assert self.fc6.get_shape().as_list()[1:] == [4096]
            self.relu6 = tf.nn.relu(self.fc6)

            self.fc7 = self._fc_layer(self.relu6, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)

            self.fc8 = self._fc_layer(self.relu7, "fc8")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

            self.data_dict = None

    def _avg_pool(self, bottom, name):
        return tf.nn._avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool2d(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            filt = self._get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self._get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _no_activation_conv_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            filt = self._get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self._get_bias(name)
            x = tf.nn.bias_add(conv, conv_biases)

            return x

    def _fc_layer(self, bottom, name):
        with tf.compat.v1.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self._get_fc_weight(name)
            biases = self._get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def _get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def _get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def _get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")
