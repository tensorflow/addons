# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements TCN layer."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Addons')
class ResidualBlock(tf.keras.layers.Layer):
    """Defines the residual block for the WaveNet TCN.

    Arguments:
        dilation_rate (int): The dilation power of 2 we are using
            for this residual block. Defaults to 1.
        filters (int): The number of convolutional
            filters to use in this block. Defaults to 64.
        kernel_size (int): The size of the convolutional kernel. Defaults
            to 2.
        padding (String): The padding used in the convolutional layers,
            'same' or 'causal'. Defaults to 'same'
        activation (String): The final activation used
            in o = Activation(x + F(x)). Defaults to 'relu'
        dropout_rate (Float): Float between 0 and 1. Fraction
            of the input units to drop. Defaults to 0.0.
        kernel_initializer (String): Initializer for the kernel weights
            matrix (Conv1D). Defaults to 'he_normal'
        use_batch_norm (bool): Whether to use batch normalization in the
            residual layers or not. Defaults to False.
        last_block (bool): Whether or not this block is the last residual
            block of the network. Defaults to False.
        kwargs: Any initializers for Layer class.

    Returns:
        A Residual Blcok.
    """

    def __init__(self,
                 dilation_rate=1,
                 filters=64,
                 kernel_size=2,
                 padding='same',
                 activation='relu',
                 dropout_rate=0.0,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 last_block=False,
                 **kwargs):

        self.dilation_rate = dilation_rate
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block

        super(ResidualBlock, self).__init__(**kwargs)

        self.residual_layers = list()

        with tf.name_scope(self.name):
            for k in range(2):
                name = 'con1D_{}'.format(k)

                with tf.name_scope(name):
                    conv_layer = tf.keras.layers.Conv1D(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        name=name,
                        kernel_initializer=self.kernel_initializer)
                    self.residual_layers.append(conv_layer)

                if self.use_batch_norm:
                    batch_norm_layer = tf.keras.layers.BatchNormalization()
                    self.residual_layers.append(batch_norm_layer)

                self.residual_layers.append(tf.keras.layers.Activation('relu'))
                self.residual_layers.append(
                    tf.keras.layers.SpatialDropout1D(
                        rate=self.dropout_rate))

            if not self.last_block:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'conv1D_{}'.format(k + 1)
                with tf.name_scope(name):
                    self.shape_match_conv = tf.keras.layers.Conv1D(
                        filters=self.filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer)

            else:
                self.shape_match_conv = tf.keras.layers.Lambda(
                    lambda x: x, name='identity')

            self.final_activation = tf.keras.layers.Activation(self.activation)

    def build(self, input_shape):

        # build residual layers
        self.res_output_shape = input_shape
        for layer in self.residual_layers:
            layer.build(self.res_output_shape)
            self.res_output_shape = layer.compute_output_shape(
                self.res_output_shape)

        # build shape matching convolutional layer
        self.shape_match_conv.build(input_shape)
        self.res_output_shape = self.shape_match_conv.compute_output_shape(
            input_shape)

        # build final activation layer
        self.final_activation.build(self.res_output_shape)

        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor,
                 and the second is the skip connection tensor.
        """
        x = inputs
        for layer in self.residual_layers:
            x = layer(x, training=training)

        x2 = self.shape_match_conv(inputs)
        res_x = x2 + x
        return [self.final_activation(res_x), x]

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]

    def get_config(self):
        config = dict()

        config['dilation_rate'] = self.dilation_rate
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['padding'] = self.padding
        config['activation'] = self.activation
        config['dropout_rate'] = self.dropout_rate
        config['use_batch_norm'] = self.use_batch_norm
        config['kernel_initializer'] = self.kernel_initializer
        config['last_block'] = self.last_block

        base_config = super(ResidualBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Addons')
class TCN(tf.keras.layers.Layer):
    """Creates a TCN layer.

    Input shape:
        A tensor of shape (batch_size, timesteps, input_dim).

    Arguments:
        filters: The number of filters to use in the convolutional layers.
            Defaults to 64.
        kernel_size: The size of the kernel to use in each
            convolutional layer. Defaults to 2.
        dilations: The array-like input of the dilations.
            Defaults to [1, 2, 4, 8, 16, 32, 64]
        stacks : The number of stacks of residual blocks to use. Defaults
            to 1.
        padding: The padding to use in the convolutional layers,
            'causal' or 'same'. Defaults to 'causal'.
        use_skip_connections: Boolean. If we want to add skip
            connections from input to each residual block.
            Defaults to True.
        return_sequences: Boolean. Whether to return the full sequence
            (when True) or the last output in the output sequence (when False).
            Defaults to False.
        activation: The activation used in the residual
            blocks o = Activation(x + F(x)). Defaults to 'linear'
        dropout_rate: Float between 0 and 1. Fraction of the input
            units to drop. Defaults to 0.0.
        kernel_initializer: Initializer for the kernel weights
            matrix (Conv1D). Defaulst to 'he_normal'
        use_batch_norm: Whether to use batch normalization in the
            residual layers or not. Defaulst to False.
        kwargs: Any other arguments for configuring parent class Layer.
            For example "name=str", Name of the model.
            Use unique names when using multiple TCN.
    Returns:
        A TCN layer.
    """

    def __init__(self,
                 filters=64,
                 kernel_size=2,
                 stacks=1,
                 dilations=[1, 2, 4, 8, 16, 32, 64],
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='linear',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 **kwargs):

        super(TCN, self).__init__(**kwargs)

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.filters = filters
        self.dilations = dilations
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm

        # validate paddings
        validate_paddings = ['causal', 'same']
        if padding not in validate_paddings:
            raise ValueError(
                "Only 'causal' or 'same' padding are compatible for this layer"
            )

        self.main_conv1D = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer)

        # list to hold all the member ResidualBlocks
        self.residual_blocks = list()
        total_num_blocks = self.stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for _ in range(self.stacks):
            for d in self.dilations:
                self.residual_blocks.append(
                    ResidualBlock(
                        dilation_rate=d,
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=self.use_batch_norm,
                        kernel_initializer=self.kernel_initializer,
                        last_block=len(self.residual_blocks) +
                        1 == total_num_blocks,
                        name='residual_block_{}'.format(
                            len(self.residual_blocks))))

        if not self.return_sequences:
            self.last_output_layer = tf.keras.layers.Lambda(
                lambda tt: tt[:, -1, :])

    def build(self, input_shape):
        self.main_conv1D.build(input_shape)

        self.build_output_shape = self.main_conv1D.compute_output_shape(
            input_shape)

        for residual_block in self.residual_blocks:
            residual_block.build(self.build_output_shape)
            self.build_output_shape = residual_block.res_output_shape

    def compute_output_shape(self, input_shape):
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            return self.last_output_layer.compute_output_shape(
                self.build_output_shape)
        else:
            return self.build_output_shape

    def call(self, inputs, training=None):
        x = inputs
        x = self.main_conv1D(x)
        skip_connections = list()
        for layer in self.residual_blocks:
            x, skip_out = layer(x, training=training)
            skip_connections.append(skip_out)

        if self.use_skip_connections:
            x = tf.keras.layers.add(skip_connections)
        if not self.return_sequences:
            x = self.last_output_layer(x)
        return x

    def get_config(self):
        config = dict()
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['stacks'] = self.stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['kernel_initializer'] = self.kernel_initializer

        base_config = super(TCN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
