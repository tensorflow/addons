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
"""Implements ESN Cell."""

import tensorflow as tf
import tensorflow.keras as keras
from typeguard import typechecked

from tensorflow_addons.utils.types import (
    Activation,
    Initializer,
)


@tf.keras.utils.register_keras_serializable(package="Addons")
class ESNCell(keras.layers.AbstractRNNCell):
    """Echo State recurrent Network (ESN) cell.
    This implements the recurrent cell from the paper:
        H. Jaeger
        "The "echo state" approach to analysing and training recurrent neural networks".
        GMD Report148, German National Research Center for Information Technology, 2001.
        https://www.researchgate.net/publication/215385037
    Arguments:
        units: Positive integer, dimensionality in the reservoir.
        connectivity: Float between 0 and 1.
            Connection probability between two reservoir units.
            Default: 0.1.
        leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it is the special case the model does not have leaky
            integration.
            Default: 1.
        spectral_radius: Float between 0 and 1.
            Desired spectral radius of recurrent weight matrix.
            Default: 0.9.
        use_norm2: Boolean, whether to use the p-norm function (with p=2) as an upper
            bound of the spectral radius so that the echo state property is satisfied.
            It  avoids to compute the eigenvalues which has an exponential complexity.
            Default: False.
        use_bias: Boolean, whether the layer uses a bias vector.
            Default: True.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            Default: `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
            Default: `glorot_uniform`.
        bias_initializer: Initializer for the bias vector.
            Default: `zeros`.
    Call arguments:
        inputs: A 2D tensor (batch x num_units).
        states: List of state tensors corresponding to the previous timestep.
    """

    @typechecked
    def __init__(
        self,
        units: int,
        connectivity: float = 0.1,
        leaky: float = 1,
        spectral_radius: float = 0.9,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation: Activation = "tanh",
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "glorot_uniform",
        bias_initializer: Initializer = "zeros",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.connectivity = connectivity
        self.leaky = leaky
        self.spectral_radius = spectral_radius
        self.use_norm2 = use_norm2
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self._state_size = units
        self._output_size = units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(tf.TensorShape(inputs_shape)[-1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]. Shape received is %s"
                % inputs_shape
            )

        def _esn_recurrent_initializer(shape, dtype, partition_info=None):
            recurrent_weights = tf.keras.initializers.get(self.recurrent_initializer)(
                shape, dtype
            )

            connectivity_mask = tf.cast(
                tf.math.less_equal(tf.random.uniform(shape), self.connectivity,), dtype
            )
            recurrent_weights = tf.math.multiply(recurrent_weights, connectivity_mask)

            # Satisfy the necessary condition for the echo state property `max(eig(W)) < 1`
            if self.use_norm2:
                # This condition is approximated scaling the norm 2 of the reservoir matrix
                # which is an upper bound of the spectral radius.
                recurrent_norm2 = tf.math.sqrt(
                    tf.math.reduce_sum(tf.math.square(recurrent_weights))
                )
                is_norm2_0 = tf.cast(tf.math.equal(recurrent_norm2, 0), dtype)
                scaling_factor = self.spectral_radius / (
                    recurrent_norm2 + 1 * is_norm2_0
                )
            else:
                abs_eig_values = tf.abs(tf.linalg.eig(recurrent_weights)[0])
                scaling_factor = tf.math.divide_no_nan(
                    self.spectral_radius, tf.reduce_max(abs_eig_values)
                )

            recurrent_weights = tf.multiply(recurrent_weights, scaling_factor)

            return recurrent_weights

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.units, self.units],
            initializer=_esn_recurrent_initializer,
            trainable=False,
            dtype=self.dtype,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self.units],
            initializer=self.kernel_initializer,
            trainable=False,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                trainable=False,
                dtype=self.dtype,
            )

        self.built = True

    def call(self, inputs, state):
        in_matrix = tf.concat([inputs, state[0]], axis=1)
        weights_matrix = tf.concat([self.kernel, self.recurrent_kernel], axis=0)

        output = tf.linalg.matmul(in_matrix, weights_matrix)
        if self.use_bias:
            output = output + self.bias
        output = self.activation(output)
        output = (1 - self.leaky) * state[0] + self.leaky * output

        return output, output

    def get_config(self):
        config = {
            "units": self.units,
            "connectivity": self.connectivity,
            "leaky": self.leaky,
            "spectral_radius": self.spectral_radius,
            "use_norm2": self.use_norm2,
            "use_bias": self.use_bias,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}
