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
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class WeightNormalization(tf.keras.layers.Wrapper):
    """This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.

    This speeds up convergence by improving the
    conditioning of the optimization problem.
    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)
    WeightNormalization wrapper works for keras and tf layers.
    ```python
      net = WeightNormalization(
          tf.keras.layers.Conv2D(2, 2, activation='relu'),
          input_shape=(32, 32, 3),
          data_init=True)(x)
      net = WeightNormalization(
          tf.keras.layers.Conv2D(16, 5, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(120, activation='relu'),
          data_init=True)(net)
      net = WeightNormalization(
          tf.keras.layers.Dense(n_classes),
          data_init=True)(net)
    ```
    Arguments:
      layer: a layer instance.
      data_init: If `True` use data dependent variable initialization
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """

    def __init__(self, layer, data_init=True, **kwargs):
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `WeightNormalization` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        self.initialized = True
        if data_init:
            self.initialized = False

        super(WeightNormalization, self).__init__(layer, **kwargs)
        self._track_trackable(layer, name='layer')

    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector with
        its norm."""
        with tf.name_scope('compute_weights'):
            self.layer.kernel = tf.nn.l2_normalize(
                self.layer.v, axis=self.kernel_norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            flat = tf.reshape(weights, [-1, self.layer_depth])
            return tf.reshape(
                tf.linalg.norm(flat, axis=0), (self.layer_depth,))

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""

        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer.call(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.layer.g = self.layer.g * scale_init
        self.layer.bias = (-m_init * scale_init)
        self.layer.activation = activation
        self.initialized = True

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`WeightNormalization` must wrap a layer that'
                                 ' contains a `kernel` for weights')

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.kernel_norm_axes = list(
                range(self.layer.kernel.shape.rank - 1))

            self.layer.v = self.layer.kernel
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=tf.keras.initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True,
                aggregation=tf.VariableAggregation.MEAN)

            # TODO: Check if this needs control deps in TF2 graph mode
            self.layer.g.assign(self._init_norm(self.layer.v))
            self._compute_weights()

            self.layer.built = True

        super(WeightNormalization, self).build()
        self.built = True

    @tf.function
    def call(self, inputs):
        """Call `Layer`"""
        if not self.initialized:
            self._data_dep_init(inputs)

        self._compute_weights()  # Recompute weights for each forward pass
        output = self.layer.call(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
