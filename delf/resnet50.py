from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging
import tensorflow as tf

layers = tf.keras.layers


def gem(x, axis=None, power=3., eps=1e-6):
    """Метод для generalized mean pooling (GeM). (використовується в реалізації tensorflow)
    """
    if axis is None:
        axis = [1, 2]
        tmp = tf.pow(tf.maximum(x, eps), power)
        out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
        return out


class _IdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, data_format):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
            filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
    def __init__(self,
                 kernel_size,
                 filters,
                 stage,
                 block,
                 data_format,
                 strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
            filters1, (1, 1),
            strides=strides,
            name=conv_name_base + '2a',
            data_format=data_format)
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
            filters2,
            kernel_size,
            padding='same',
            name=conv_name_base + '2b',
            data_format=data_format)
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
            filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2c')

        self.conv_shortcut = layers.Conv2D(
            filters3, (1, 1),
            strides=strides,
            name=conv_name_base + '1',
            data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


class ResNet50(tf.keras.Model):
    def __init__(self,
                 data_format,
                 name='',
                 include_top=True,
                 pooling=None,
                 block3_strides=False,
                 average_pooling=True,
                 classes=1000,
                 gem_power=3.0,
                 embedding_layer=False,
                 embedding_layer_dim=2048):

        super(ResNet50, self).__init__(name=name)

        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' %
                             (data_format, valid_channel_values))
        self.include_top = include_top
        self.block3_strides = block3_strides
        self.average_pooling = average_pooling
        self.pooling = pooling

        def conv_block(filters, stage, block, strides=(2, 2)):
            return _ConvBlock(
                3,
                filters,
                stage=stage,
                block=block,
                data_format=data_format,
                strides=strides)

        def id_block(filters, stage, block):
            return _IdentityBlock(
                3, filters, stage=stage, block=block, data_format=data_format)

        self.conv1 = layers.Conv2D(
            64, (7, 7),
            strides=(2, 2),
            data_format=data_format,
            padding='same',
            name='conv1')
        bn_axis = 1 if data_format == 'channels_first' else 3
        self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D((3, 3),
                                            strides=(2, 2),
                                            data_format=data_format)

        self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.l2b = id_block([64, 64, 256], stage=2, block='b')
        self.l2c = id_block([64, 64, 256], stage=2, block='c')

        self.l3a = conv_block([128, 128, 512], stage=3, block='a')
        self.l3b = id_block([128, 128, 512], stage=3, block='b')
        self.l3c = id_block([128, 128, 512], stage=3, block='c')
        self.l3d = id_block([128, 128, 512], stage=3, block='d')

        self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
        self.l4b = id_block([256, 256, 1024], stage=4, block='b')
        self.l4c = id_block([256, 256, 1024], stage=4, block='c')
        self.l4d = id_block([256, 256, 1024], stage=4, block='d')
        self.l4e = id_block([256, 256, 1024], stage=4, block='e')
        self.l4f = id_block([256, 256, 1024], stage=4, block='f')

        # Striding layer that can be used on top of block3 to produce feature maps
        # with the same resolution as the TF-Slim implementation.
        if self.block3_strides:
            self.subsampling_layer = layers.MaxPooling2D((1, 1),
                                                         strides=(2, 2),
                                                         data_format=data_format)
            self.l5a = conv_block([512, 512, 2048],
                                  stage=5,
                                  block='a',
                                  strides=(1, 1))
        else:
            self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
        self.l5b = id_block([512, 512, 2048], stage=5, block='b')
        self.l5c = id_block([512, 512, 2048], stage=5, block='c')

        self.avg_pool = layers.AveragePooling2D((7, 7),
                                                strides=(7, 7),
                                                data_format=data_format)

        if self.include_top:
            self.flatten = layers.Flatten()
            self.fc1000 = layers.Dense(classes, name='fc1000')
        else:
            reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
            reduction_indices = tf.constant(reduction_indices)
            if pooling == 'avg':
                self.global_pooling = functools.partial(
                    tf.reduce_mean, axis=reduction_indices, keepdims=False)
            elif pooling == 'max':
                self.global_pooling = functools.partial(
                    tf.reduce_max, axis=reduction_indices, keepdims=False)
            elif pooling == 'gem':
                logging.info('Adding GeMPooling layer with power %f', gem_power)
                self.global_pooling = functools.partial(
                    gem, axis=reduction_indices, power=gem_power)
            else:
                self.global_pooling = None
            if embedding_layer:
                logging.info('Adding embedding layer with dimension %d',
                             embedding_layer_dim)
                self.embedding_layer = layers.Dense(
                    embedding_layer_dim, name='embedding_layer')
            else:
                self.embedding_layer = None

    def build_call(self, inputs, training=True, intermediates_dict=None):
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        if intermediates_dict is not None:
            intermediates_dict['block0'] = x

        x = self.max_pool(x)
        if intermediates_dict is not None:
            intermediates_dict['block0mp'] = x

        # Block 1 (equivalent to "conv2" in Resnet paper).
        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)
        if intermediates_dict is not None:
            intermediates_dict['block1'] = x

        # Block 2 (equivalent to "conv3" in Resnet paper).
        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)
        if intermediates_dict is not None:
            intermediates_dict['block2'] = x

        # Block 3 (equivalent to "conv4" in Resnet paper).
        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)

        if self.block3_strides:
            x = self.subsampling_layer(x)
            if intermediates_dict is not None:
                intermediates_dict['block3'] = x
        else:
            if intermediates_dict is not None:
                intermediates_dict['block3'] = x

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)

        if self.average_pooling:
            x = self.avg_pool(x)
            if intermediates_dict is not None:
                intermediates_dict['block4'] = x
        else:
            if intermediates_dict is not None:
                intermediates_dict['block4'] = x

        if self.include_top:
            return self.fc1000(self.flatten(x))
        elif self.global_pooling:
            x = self.global_pooling(x)
        if self.embedding_layer:
            x = self.embedding_layer(x)
            return x
        else:
            return x

    def call(self, inputs, training=True, intermediates_dict=None):
        return self.build_call(inputs, training, intermediates_dict)