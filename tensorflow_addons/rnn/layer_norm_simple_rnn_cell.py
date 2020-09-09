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
"""Implements LayerNormSimpleRNNCell Cell."""

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
class LayerNormSimpleRNNCell(keras.layers.SimpleRNNCell):
    """Cell class for LayerNormSimpleRNN.

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
        "Layer Normalization." ArXiv:1607.06450 [Cs, Stat],
        July 21, 2016. http://arxiv.org/abs/1607.06450

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias
        vector.
      layernorm_epsilon: Float, (default `1e-5`), Small float added to variance
        to avoid dividing by zero.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector (`use_bias=True`).
         Default: `zeros`.
      gamma_initializer: Initializer for the gamma vector of the layer
         normalization layer. Default: `ones`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector
         (`use_bias=True`). Default: `None`.
      gamma_regularizer: Regularizer function applied to the gamma vector
         of the layer normalization layer. Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector
         (`use_bias=True`). Default: `None`.
      gamma_constraint: Constraint function applied to the gamma vector
         of the layer normalization layer. Default: `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state. Default: 0.

    Call arguments:
      inputs: A 2D tensor, with shape of `[batch, feature]`.
      states: A 2D tensor with shape of `[batch, units]`, which is the state
        from the previous time step. For timestep 0, the initial state provided
        by the user will be feed to cell.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.

    Examples:

    ```python
    import numpy as np
    import tensorflow.keras as keras
    import tensorflow_addons as tfa

    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    rnn = keras.layers.RNN(tfa.rnn.LayerNormSimpleRNNCell(4))

    output = rnn(inputs)  # The output has shape `[32, 4]`.

    rnn = keras.layers.RNN(
        tfa.rnn.LayerNormSimpleRNNCell(4),
        return_sequences=True,
        return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = rnn(inputs)
    ```
    """

    @typechecked
    def __init__(
        self,
        units: TensorLike,
        activation: Activation = "tanh",
        use_bias: bool = True,
        layernorm_epsilon: FloatTensorLike = 1e-05,
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "orthogonal",
        bias_initializer: Initializer = "zeros",
        gamma_initializer: Initializer = "ones",
        kernel_regularizer: Regularizer = None,
        recurrent_regularizer: Regularizer = None,
        bias_regularizer: Regularizer = None,
        gamma_regularizer: Regularizer = None,
        kernel_constraint: Regularizer = None,
        recurrent_constraint: Constraint = None,
        bias_constraint: Constraint = None,
        gamma_constraint: Constraint = None,
        dropout: FloatTensorLike = 0.0,
        recurrent_dropout: FloatTensorLike = 0.0,
        **kwargs
    ):
        super(LayerNormSimpleRNNCell, self).__init__(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
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
        self.layernorm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=layernorm_epsilon,
            center=False,
            scale=True,
            beta_initializer=None,
            gamma_initializer=gamma_initializer,
            beta_regularizer=None,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=None,
            gamma_constraint=gamma_constraint,
            **kwargs,
        )

    def build(self, input_shape):
        super(LayerNormSimpleRNNCell, self).build(input_shape)
        self.layernorm.build((None, self.units))

    def call(self, inputs, states, training=None):
        """Formulas.

        Notation:
            y_t : Cell output at t (`output`)
            y_{t-1} : Previous cell output at t-1 (`prev_output`)
            x_t : The new input at t (`inputs`)
            W_xh : Weight matrix for inputs x_t (`self.kernel`)
            W_hh : Weights for prev. outputs y_{t-1} (`self.recurrent_kernel`)
            b : Bias term for centering (`self.bias`)
            d1 : Dropout function for x_t (`inputs * dp_mask`)
            d2 : Dropout function for y_{t-1} (`prev_output * rec_dp_mask`)
            ln : Scaling function from layer normalization (`self.layernorm`)
            f : Activation function (`self.activation`)

        Case 1:
            Keras' SimpleRNN. Only with bias and activation
              y_t = f(x_t * W_xh + y_{t-1} * W_hh + b)
            or
              net = x_t * W_xh + y_{t-1} * W_hh
              y_t = f(net + b)

        Case 2:
            addons' LayerNormSimpleRNNCell. Like case 1 but with layer
            normalization (only scaling).
              y_t = f(ln(x_t * W_xh + y_{t-1} * W_hh) + b)
            or
              net = x_t * W_xh + y_{t-1} * W_hh
              y_t = f(ln(net) + b)

            Layer normalization with scaling and centering in one go (see Ba et
            al (2016), page 3, formula 4, https://arxiv.org/abs/1607.06450)
            is the same as layer normalization only with scaling, and
            centering directly afterwards.

        Case 3:
            Keras' SimpleRNN. with dropout, bias, and activation
              y_t = f(d1(x_t) * W_xh + d2(y_{t-1}) * W_hh + b)
            or
              net = d1(x_t) * W_xh + d2(y_{t-1}) * W_hh
              y_t = f(net + b)

        Case 4:
            addons' LayerNormSimpleRNNCell. Like case 3 but with layer
            normalization (only scaling).
              y_t = f(ln(d1(x_t) * W_xh + d2(y_{t-1}) * W_hh) + b)
            or
              net = d1(x_t) * W_xh + d2(y_{t-1}) * W_hh
              y_t = f(ln(net) + b)
        """
        prev_output = states[0]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)

        if dp_mask is not None:
            h = keras.backend.dot(inputs * dp_mask, self.kernel)
        else:
            h = keras.backend.dot(inputs, self.kernel)

        # don't add bias to "h" here
        # add bias after scaling with layer normalization to "output"

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + keras.backend.dot(prev_output, self.recurrent_kernel)  # "net"

        output = self.layernorm(output)

        if self.bias is not None:
            output = keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output, [output]

    # use SimpleRNNCell's get_initial_state method

    def get_config(self):
        cell_config = super(LayerNormSimpleRNNCell, self).get_config()
        del cell_config["name"]

        ln_config = self.layernorm.get_config()
        ln_config = {
            k: v
            for k, v in ln_config.items()
            if k
            in ["epsilon", "gamma_initializer", "gamma_regularizer", "gamma_constraint"]
        }

        ln_config["layernorm_epsilon"] = ln_config.pop("epsilon")
        return dict(list(cell_config.items()) + list(ln_config.items()))
