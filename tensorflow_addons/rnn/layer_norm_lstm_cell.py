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
"""Implements LayerNormLSTM Cell."""

import tensorflow as tf
import tensorflow.keras as keras
from typeguard import typechecked

from tensorflow_addons.utils.types import (
    Activation,
    FloatTensorLike,
    TensorLike,
    Initializer,
    Constraint,
    Regularizer,
)


@tf.keras.utils.register_keras_serializable(package="Addons")
class LayerNormLSTMCell(keras.layers.LSTMCell):
    """LSTM cell with layer normalization and recurrent dropout.

    This class adds layer normalization and recurrent dropout to a LSTM unit.
    Layer normalization implementation is based on:

      https://arxiv.org/abs/1607.06450.

    "Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

    and is applied before the internal nonlinearities.
    Recurrent dropout is based on:

      https://arxiv.org/abs/1603.05118

    "Recurrent Dropout without Memory Loss"
    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.

    Example:

    >>> inputs = np.random.random([30,23,9]).astype(np.float32)
    >>> lnLSTMCell = tfa.rnn.LayerNormLSTMCell(4)
    >>> rnn = tf.keras.layers.RNN(lnLSTMCell, return_sequences=True, return_state=True)
    >>> outputs, memory_state, carry_state = rnn(inputs)
    >>> outputs.shape
    TensorShape([30, 23, 4])
    >>> memory_state.shape
    TensorShape([30, 4])
    >>> carry_state.shape
    TensorShape([30, 4])
    """

    @typechecked
    def __init__(
        self,
        units: TensorLike,
        activation: Activation = "tanh",
        recurrent_activation: Activation = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "orthogonal",
        bias_initializer: Initializer = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer: Regularizer = None,
        recurrent_regularizer: Regularizer = None,
        bias_regularizer: Regularizer = None,
        kernel_constraint: Constraint = None,
        recurrent_constraint: Constraint = None,
        bias_constraint: Constraint = None,
        dropout: FloatTensorLike = 0.0,
        recurrent_dropout: FloatTensorLike = 0.0,
        norm_gamma_initializer: Initializer = "ones",
        norm_beta_initializer: Initializer = "zeros",
        norm_epsilon: FloatTensorLike = 1e-3,
        **kwargs,
    ):
        """Initializes the LSTM cell.

        Args:
          units: Positive integer, dimensionality of the output space.
          activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass `None`, no activation is applied (ie.
            "linear" activation: `a(x) = x`).
          recurrent_activation: Activation function to use for the recurrent
            step. Default: sigmoid (`sigmoid`). If you pass `None`, no
            activation is applied (ie. "linear" activation: `a(x) = x`).
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix, used
            for the linear transformation of the inputs.
          recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
          bias_initializer: Initializer for the bias vector.
          unit_forget_bias: Boolean. If True, add 1 to the bias of the forget
            gate at initialization. Setting it to true will also force
            `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
              al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
          kernel_regularizer: Regularizer function applied to the `kernel`
            weights matrix.
          recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          kernel_constraint: Constraint function applied to the `kernel`
            weights matrix.
          recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.
          dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.
          recurrent_dropout: Float between 0 and 1. Fraction of the units to
            drop for the linear transformation of the recurrent state.
          norm_gamma_initializer: Initializer for the layer normalization gain
            initial value.
          norm_beta_initializer: Initializer for the layer normalization shift
            initial value.
          norm_epsilon: Float, the epsilon value for normalization layers.
          **kwargs: Dict, the other keyword arguments for layer creation.
        """
        super().__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs,
        )
        self.norm_gamma_initializer = keras.initializers.get(norm_gamma_initializer)
        self.norm_beta_initializer = keras.initializers.get(norm_beta_initializer)
        self.norm_epsilon = norm_epsilon
        self.kernel_norm = self._create_norm_layer("kernel_norm")
        self.recurrent_norm = self._create_norm_layer("recurrent_norm")
        self.state_norm = self._create_norm_layer("state_norm")

    def build(self, input_shape):
        super().build(input_shape)

        def maybe_build_sublayer(sublayer, build_shape):
            if not sublayer.built:
                with tf.keras.backend.name_scope(sublayer.name):
                    sublayer.build(build_shape)
                    sublayer.built = True

        maybe_build_sublayer(self.kernel_norm, [input_shape[0], self.units * 4])
        maybe_build_sublayer(self.recurrent_norm, [input_shape[0], self.units * 4])
        maybe_build_sublayer(self.state_norm, [input_shape[0], self.units])

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)
        if 0.0 < self.dropout < 1.0:
            inputs *= dp_mask[0]
        z = self.kernel_norm(keras.backend.dot(inputs, self.kernel))

        if 0.0 < self.recurrent_dropout < 1.0:
            h_tm1 *= rec_dp_mask[0]
        z += self.recurrent_norm(keras.backend.dot(h_tm1, self.recurrent_kernel))
        if self.use_bias:
            z = keras.backend.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        c = self.state_norm(c)
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {
            "norm_gamma_initializer": keras.initializers.serialize(
                self.norm_gamma_initializer
            ),
            "norm_beta_initializer": keras.initializers.serialize(
                self.norm_beta_initializer
            ),
            "norm_epsilon": self.norm_epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_norm_layer(self, name):
        return keras.layers.LayerNormalization(
            beta_initializer=self.norm_beta_initializer,
            gamma_initializer=self.norm_gamma_initializer,
            epsilon=self.norm_epsilon,
            name=name,
        )
