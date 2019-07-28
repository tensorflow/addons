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
        super(WeightNormalization, self).__init__(layer, **kwargs)
        self.data_init = data_init
        self._initialized = False
        self._track_trackable(layer, name='layer')

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, 'kernel'):
                raise ValueError('`WeightNormalization` must wrap a layer that'
                                 ' contains a `kernel` for weights')

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.kernel_norm_axes = list(
                range(self.layer.kernel.shape.rank - 1))

            self.v = self.layer.kernel
            self.g = self.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=tf.keras.initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

        super(WeightNormalization, self).build()

    @tf.function
    def call(self, inputs):
        """Call `Layer`"""
        if not self._initialized:
            self._initialize_weights(inputs)

        self._compute_weights()  # Recompute weights for each forward pass
        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _compute_weights(self):
        """Generate normalized weights.

        This method will update the value of self.layer.kernel with the
        normalized value, so that the layer is ready for call().
        """
        with tf.name_scope('compute_weights'):
            self.layer.kernel = tf.nn.l2_normalize(
                self.v, axis=self.kernel_norm_axes) * self.g

    def _initialize_weights(self, inputs):
        """Initialize weight g.

        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        if self.data_init:
            self._data_dep_init(inputs)
        else:
            self._init_norm()
        self._initialized = True

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope('init_norm'):
            flat = tf.reshape(self.v, [-1, self.layer_depth])
            self.g.assign(
                tf.reshape(tf.linalg.norm(flat, axis=0), (self.layer_depth,)))

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""

        with tf.name_scope('data_dep_init'):
            # Generate data dependent init values
            existing_activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1. / tf.math.sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.g = self.g * scale_init
        if hasattr(self.layer, 'bias'):
            self.layer.bias = -m_init * scale_init
        self.layer.activation = existing_activation

    def get_config(self):
        config = {'data_init': self.data_init}
        base_config = super(WeightNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_utils.register_keras_custom_object
class SpectralNormalization(tf.keras.layers.Wrapper):
    """This wrapper controls the Lipschitz constant of the layer by
       constraining its spectral norm.

    This speeds up convergence by improving the conditioning of the
    optimization problem.

    [1] Spectral Normalization for Generative Adversarial Networks:
    https://arxiv.org/abs/1802.05957
    Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida (2018)

    SpectralNormalization wrapper works for keras and tf layers.
    ```python
      net = SpectralNormalization(
          tf.keras.layers.Conv2D(2, 2, activation="relu"),
          input_shape=(32, 32, 3))(x)
      net = SpectralNormalization(
          tf.keras.layers.Conv2D(16, 5, activation="relu"))(net)
      net = SpectralNormalization(
          tf.keras.layers.Dense(120, activation="relu"))(net)
      net = SpectralNormalization(
          tf.keras.layers.Dense(n_classes))(net)
    ```
    Arguments:
      layer: a layer instance.
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of spectrals
      NotImplementedError: If `data_init` is True and running graph execution
    """

    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        assert power_iterations > 0, (
            "`power_iterations` should be positive, got `power_iterations={}`"
            "".format(power_iterations))
        self.power_iterations = power_iterations
        self._initialized = False
        self._track_trackable(layer, name="layer")

    def build(self, input_shape):
        """Build `Layer`"""

        input_shape = tf.TensorShape(input_shape).as_list()
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

            if not hasattr(self.layer, "kernel"):
                raise ValueError(
                    "`SpectralNormalization` must wrap a layer that contains"
                    " a `kernel` attribute for parameters.")

            self.u = self.add_variable(
                name="u",
                shape=(1, int(self.layer.kernel.shape[-1])),
                initializer=tf.keras.initializers.get("normal"),
                dtype=self.layer.kernel.dtype,
                trainable=False)

        super(SpectralNormalization, self).build()

    @tf.function
    def call(self, *args, **kwargs):
        """Call `Layer`"""
        self._spectral_normalize()
        output = self.layer(*args, **kwargs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

    def _spectral_normalize(self):
        """Generate spectral normalized weights.

        This method will update the value of self.layer.kernel with the
        spectral normalized value, so that the layer is ready for call().
        """

        w = self.layer.kernel
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for i in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))

            self.layer.kernel = w / sigma
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super(SpectralNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
