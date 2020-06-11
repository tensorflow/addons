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
"""Module for RNN Cells."""

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
from typing import Optional


@tf.keras.utils.register_keras_serializable(package="Addons")
class NASCell(keras.layers.AbstractRNNCell):
    """Neural Architecture Search (NAS) recurrent network cell.

    This implements the recurrent cell from the paper:

      https://arxiv.org/abs/1611.01578

    Barret Zoph and Quoc V. Le.
    "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.

    The class uses an optional projection layer.
    """

    # NAS cell's architecture base.
    _NAS_BASE = 8

    @typechecked
    def __init__(
        self,
        units: TensorLike,
        projection: Optional[FloatTensorLike] = None,
        use_bias: bool = False,
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "glorot_uniform",
        projection_initializer: Initializer = "glorot_uniform",
        bias_initializer: Initializer = "zeros",
        **kwargs
    ):
        """Initialize the parameters for a NAS cell.

        Args:
          units: int, The number of units in the NAS cell.
          projection: (optional) int, The output dimensionality for the
            projection matrices.  If None, no projection is performed.
          use_bias: (optional) bool, If True then use biases within the cell.
            This is False by default.
          kernel_initializer: Initializer for kernel weight.
          recurrent_initializer: Initializer for recurrent kernel weight.
          projection_initializer: Initializer for projection weight, used when
            projection is not None.
          bias_initializer: Initializer for bias, used when use_bias is True.
          **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.units = units
        self.projection = projection
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.projection_initializer = projection_initializer
        self.bias_initializer = bias_initializer

        if projection is not None:
            self._state_size = [units, projection]
            self._output_size = projection
        else:
            self._state_size = [units, units]
            self._output_size = units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        input_size = tf.compat.dimension_value(
            tf.TensorShape(inputs_shape).with_rank(2)[1]
        )
        if input_size is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # Variables for the NAS cell. `recurrent_kernel` is all matrices
        # multiplying the hidden state and `kernel` is all matrices multiplying
        # the inputs.
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.output_size, self._NAS_BASE * self.units],
            initializer=self.recurrent_initializer,
        )
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self._NAS_BASE * self.units],
            initializer=self.kernel_initializer,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self._NAS_BASE * self.units],
                initializer=self.bias_initializer,
            )
        # Projection layer if specified
        if self.projection is not None:
            self.projection_weights = self.add_weight(
                name="projection_weights",
                shape=[self.units, self.projection],
                initializer=self.projection_initializer,
            )

        self.built = True

    def call(self, inputs, state):
        """Run one step of NAS Cell.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: This must be a list of state Tensors, both `2-D`, with column
            sizes `c_state` and `m_state`.

        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of
            the NAS Cell after reading `inputs` when previous state was
            `state`.
            Here output_dim is:
               projection if projection was set, units otherwise.
          - Tensor(s) representing the new state of NAS Cell after reading
            `inputs` when the previous state was `state`.  Same type and
            shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        sigmoid = tf.math.sigmoid
        tanh = tf.math.tanh
        relu = tf.nn.relu

        c_prev, m_prev = state

        m_matrix = tf.matmul(m_prev, self.recurrent_kernel)
        inputs_matrix = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            m_matrix = tf.nn.bias_add(m_matrix, self.bias)

        # The NAS cell branches into 8 different splits for both the hidden
        # state and the input
        m_matrix_splits = tf.split(
            axis=1, num_or_size_splits=self._NAS_BASE, value=m_matrix
        )
        inputs_matrix_splits = tf.split(
            axis=1, num_or_size_splits=self._NAS_BASE, value=inputs_matrix
        )

        # First layer
        layer1_0 = sigmoid(inputs_matrix_splits[0] + m_matrix_splits[0])
        layer1_1 = relu(inputs_matrix_splits[1] + m_matrix_splits[1])
        layer1_2 = sigmoid(inputs_matrix_splits[2] + m_matrix_splits[2])
        layer1_3 = relu(inputs_matrix_splits[3] * m_matrix_splits[3])
        layer1_4 = tanh(inputs_matrix_splits[4] + m_matrix_splits[4])
        layer1_5 = sigmoid(inputs_matrix_splits[5] + m_matrix_splits[5])
        layer1_6 = tanh(inputs_matrix_splits[6] + m_matrix_splits[6])
        layer1_7 = sigmoid(inputs_matrix_splits[7] + m_matrix_splits[7])

        # Second layer
        l2_0 = tanh(layer1_0 * layer1_1)
        l2_1 = tanh(layer1_2 + layer1_3)
        l2_2 = tanh(layer1_4 * layer1_5)
        l2_3 = sigmoid(layer1_6 + layer1_7)

        # Inject the cell
        l2_0 = tanh(l2_0 + c_prev)

        # Third layer
        l3_0_pre = l2_0 * l2_1
        new_c = l3_0_pre  # create new cell
        l3_0 = l3_0_pre
        l3_1 = tanh(l2_2 + l2_3)

        # Final layer
        new_m = tanh(l3_0 * l3_1)

        # Projection layer if specified
        if self.projection is not None:
            new_m = tf.matmul(new_m, self.projection_weights)

        return new_m, [new_c, new_m]

    def get_config(self):
        config = {
            "units": self.units,
            "projection": self.projection,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "recurrent_initializer": self.recurrent_initializer,
            "bias_initializer": self.bias_initializer,
            "projection_initializer": self.projection_initializer,
        }
        base_config = super().get_config()
        return {**base_config, **config}


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
        **kwargs
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
        self.kernel_norm.build([input_shape[0], self.units * 4])
        self.recurrent_norm.build([input_shape[0], self.units * 4])
        self.state_norm.build([input_shape[0], self.units])

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
