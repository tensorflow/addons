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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras


@tf.keras.utils.register_keras_serializable(package='Addons')
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

    def __init__(self,
                 units,
                 projection=None,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 recurrent_initializer="glorot_uniform",
                 projection_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 **kwargs):
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
        super(NASCell, self).__init__(**kwargs)
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
            tf.TensorShape(inputs_shape).with_rank(2)[1])
        if input_size is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")

        # Variables for the NAS cell. `recurrent_kernel` is all matrices
        # multiplying the hidden state and `kernel` is all matrices multiplying
        # the inputs.
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=[self.output_size, self._NAS_BASE * self.units],
            initializer=self.recurrent_initializer)
        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_size, self._NAS_BASE * self.units],
            initializer=self.kernel_initializer)

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self._NAS_BASE * self.units],
                initializer=self.bias_initializer)
        # Projection layer if specified
        if self.projection is not None:
            self.projection_weights = self.add_weight(
                name="projection_weights",
                shape=[self.units, self.projection],
                initializer=self.projection_initializer)

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
            axis=1, num_or_size_splits=self._NAS_BASE, value=m_matrix)
        inputs_matrix_splits = tf.split(
            axis=1, num_or_size_splits=self._NAS_BASE, value=inputs_matrix)

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
        base_config = super(NASCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Addons')
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

    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 norm_gamma_initializer='ones',
                 norm_beta_initializer='zeros',
                 norm_epsilon=1e-3,
                 **kwargs):
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
        super(LayerNormLSTMCell, self).__init__(
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
            **kwargs)
        self.norm_gamma_initializer = keras.initializers.get(
            norm_gamma_initializer)
        self.norm_beta_initializer = keras.initializers.get(
            norm_beta_initializer)
        self.norm_epsilon = norm_epsilon
        self.kernel_norm = self._create_norm_layer('kernel_norm')
        self.recurrent_norm = self._create_norm_layer('recurrent_norm')
        self.state_norm = self._create_norm_layer('state_norm')

    def build(self, input_shape):
        super(LayerNormLSTMCell, self).build(input_shape)
        self.kernel_norm.build([input_shape[0], self.units * 4])
        self.recurrent_norm.build([input_shape[0], self.units * 4])
        self.state_norm.build([input_shape[0], self.units])

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)
        if 0. < self.dropout < 1.:
            inputs *= dp_mask[0]
        z = self.kernel_norm(keras.backend.dot(inputs, self.kernel))

        if 0. < self.recurrent_dropout < 1.:
            h_tm1 *= rec_dp_mask[0]
        z += self.recurrent_norm(
            keras.backend.dot(h_tm1, self.recurrent_kernel))
        if self.use_bias:
            z = keras.backend.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        c = self.state_norm(c)
        h = o * self.activation(c)
        return h, [h, c]

    def get_config(self):
        config = {
            'norm_gamma_initializer':
            keras.initializers.serialize(self.norm_gamma_initializer),
            'norm_beta_initializer':
            keras.initializers.serialize(self.norm_beta_initializer),
            'norm_epsilon':
            self.norm_epsilon,
        }
        base_config = super(LayerNormLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_norm_layer(self, name):
        return keras.layers.LayerNormalization(
            beta_initializer=self.norm_beta_initializer,
            gamma_initializer=self.norm_gamma_initializer,
            epsilon=self.norm_epsilon,
            name=name)


@tf.keras.utils.register_keras_serializable(package='Addons')
# class LayernormSimpleRNNCell(SimpleRNNCell, LayerNormalization):
class LayernormSimpleRNNCell(keras.layers.SimpleRNNCell):
    """Cell class for LayernormSimpleRNN.

    Motivation:
    - Drop-In Replacement for keras.layers.SimpleRNNCell
    - demonstrate how to add keras.layers.LayerNormalization
       to all RNNs by introducing the `use_layernorm` argument

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
        “Layer Normalization.” ArXiv:1607.06450 [Cs, Stat],
        July 21, 2016. http://arxiv.org/abs/1607.06450

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias
        vector.
      use_layernorm: Boolean, (default `True`), whether to apply layer
        normalization (scaling only).
      layernorm_epsilon: Float, (default `1e-5`), Small float added to variance
        to avoid dividing by zero.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector (`use_bias=True`) or
         for the beta vector in layer normalization (`use_layernorm=True`).
         Default: `zeros`.
      gamma_initializer: Initializer for the gamma vector of the layer
         normalization layer (`use_layernorm=True`). Default: `ones`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_regularizer: Regularizer function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_constraint: Constraint function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
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
    rnn = keras.layers.RNN(tfa.rnn.LayernormSimpleRNNCell(4))

    output = rnn(inputs)  # The output has shape `[32, 4]`.

    rnn = keras.layers.RNN(
        tfa.rnn.LayernormSimpleRNNCell(4),
        return_sequences=True,
        return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = rnn(inputs)
    ```
    """

    def __init__(
            self,
            units,
            activation='tanh',
            use_bias=True,
            use_layernorm=True,  # NEW(!)
            layernorm_epsilon=1e-05,  # NEW(!)
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            gamma_initializer='ones',  # NEW(!)
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            gamma_regularizer=None,  # NEW(!)
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            gamma_constraint=None,  # NEW(!)
            dropout=0.,
            recurrent_dropout=0.,
            **kwargs):
        self.use_layernorm = use_layernorm
        keras.layers.SimpleRNNCell.__init__(
            self,
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
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        if use_layernorm:
            # LayerNormalization.__init__(self,
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
                dtype=kwargs.get('dtype'),
                trainable=kwargs.get('trainable', True))

    # @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # SimpleRNNCell.build(self, input_shape)
        super(LayernormSimpleRNNCell, self).build(input_shape)
        if self.use_layernorm:
            # LayerNormalization.build(self, (None, self.units))
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
            Simple RNN, only with bias and activation
              y_t = f(x_t * W_xh + y_{t-1} * W_hh + b)
            or
              net = x_t * W_xh + y_{t-1} * W_hh
              y_t = f(net + b)

        Case 2:
            RNN with, layer normalization (only scaling), bias and activation.
              y_t = f(ln(x_t * W_xh + y_{t-1} * W_hh) + b)
            or
              net = x_t * W_xh + y_{t-1} * W_hh
              y_t = f(ln(net) + b)

            Layer normalization with scaling and centering in one go (see Ba et
            al (2016), page 3, formula 4, https://arxiv.org/abs/1607.06450)
            is the same as layer normalization only with scaling, and
            centering directly afterwards.

        Case 3:
            RNN, with dropout, bias, and activation (no scaling from LN)
              y_t = f(d1(x_t) * W_xh + d2(y_{t-1}) * W_hh + b)
            or
              net = d1(x_t) * W_xh + d2(y_{t-1}) * W_hh
              y_t = f(net + b)

        Case 4:
            Everyting is used, i.e. all dropouts, layer normalization
            (only scaling), bias, and activation
              y_t = f(ln(d1(x_t) * W_xh + d2(y_{t-1}) * W_hh) + b)
            or
              net = d1(x_t) * W_xh + d2(y_{t-1}) * W_hh
              y_t = f(ln(net) + b)
        """
        prev_output = states[0]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            h = keras.backend.dot(inputs * dp_mask, self.kernel)
        else:
            h = keras.backend.dot(inputs, self.kernel)

        # don't add bias to "h" here
        # add bias after scaling with layer normalization to "output"

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + keras.backend.dot(prev_output,
                                       self.recurrent_kernel)  # "net"

        if self.use_layernorm:
            # output = LayerNormalization.call(self, output)
            output = self.layernorm(output)

        if self.bias is not None:
            output = keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output, [output]

    # use SimpleRNNCell's get_initial_state method

    def get_config(self):
        config = {'use_layernorm': self.use_layernorm}
        cell_config = keras.layers.SimpleRNNCell.get_config(self)
        del cell_config['name']
        if self.use_layernorm:
            # ln_config = LayerNormalization.get_config(self)
            ln_config = self.layernorm.get_config()
            ln_config = {
                key: ln_config[key]
                for key in [
                    "epsilon", "gamma_initializer", "gamma_regularizer",
                    "gamma_constraint"
                ] if key in ln_config
            }
            ln_config['layernorm_epsilon'] = ln_config.pop("epsilon")
        else:
            ln_config = {}
        return {**config, **cell_config, **ln_config}


@tf.keras.utils.register_keras_serializable(package='Addons')
class LayernormSimpleRNN(keras.layers.SimpleRNN):
    """Fully-connected RNN with Layer Normalization.

    Motivation:
    - Drop-In Replacement for keras.layers.SimpleRNN
    - demonstrate how to add keras.layers.LayerNormalization
       to all RNNs by introducing the `use_layernorm` argument

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
        “Layer Normalization.” ArXiv:1607.06450 [Cs, Stat],
        July 21, 2016. http://arxiv.org/abs/1607.06450

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias
        vector.
      use_layernorm: Boolean, (default `True`), whether to apply layer
        normalization (scaling only).
      layernorm_epsilon: Float, (default `1e-5`), Small float added to variance
        to avoid dividing by zero.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector (`use_bias=True`) or
         for the beta vector in layer normalization (`use_layernorm=True`).
         Default: `zeros`.
      gamma_initializer: Initializer for the gamma vector of the layer
         normalization layer (`use_layernorm=True`). Default: `ones`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_regularizer: Regularizer function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation"). Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.  Default: `None`.
      bias_constraint: Constraint function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_constraint: Constraint function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the
        inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the
        recurrent state. Default: 0.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. Default: `False`.
      return_state: Boolean. Whether to return the last state
        in addition to the output. Default: `False`
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.

    Call arguments:
      inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
      mask: Binary tensor of shape `[batch, timesteps]` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.

    Examples:

    ```python
    import numpy as np
    import tensorflow_addons as tfa

    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    model = tfa.rnn.LayernormSimpleRNN(4)

    output = model(inputs)  # The output has shape `[32, 4]`.

    model = tfa.rnn.LayernormSimpleRNN(
        4, return_sequences=True, return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = model(inputs)
    ```
    """

    def __init__(
            self,
            units,
            activation='tanh',
            use_bias=True,
            use_layernorm=True,  # NEW(!)
            layernorm_epsilon=1e-05,  # NEW(!)
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            gamma_initializer='ones',  # NEW(!)
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            gamma_regularizer=None,  # NEW(!)
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            gamma_constraint=None,  # NEW(!)
            dropout=0.,
            recurrent_dropout=0.,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            **kwargs):
        # 'implementation' warning was never relevant for LayernormSimpleRNN
        cell = LayernormSimpleRNNCell(
            units,
            activation=activation,
            use_bias=use_bias,
            use_layernorm=use_layernorm,  # NEW(!)
            layernorm_epsilon=layernorm_epsilon,  # NEW(!)
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            gamma_initializer=gamma_initializer,  # NEW(!)
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            gamma_regularizer=gamma_regularizer,  # NEW(!)
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            gamma_constraint=gamma_constraint,  # NEW(!)
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        super(keras.layers.SimpleRNN, self).__init__(  # call RNN's init
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        # IT'S NOT USED ANYWHERE(!):
        # self.activity_regularizer = regularizers.get(activity_regularizer)
        # self.input_spec = [InputSpec(ndim=3)]

    # use SimpleRNN's call() method

    @property
    def use_layernorm(self):
        return self.cell.use_layernorm

    @property
    def layernorm_epsilon(self):
        return self.cell.layernorm_epsilon

    @property
    def gamma_initializer(self):
        return self.cell.gamma_initializer

    @property
    def gamma_regularizer(self):
        return self.cell.gamma_regularizer

    @property
    def gamma_constraint(self):
        return self.cell.gamma_constraint

    def get_config(self):
        base_config = super(keras.layers.SimpleRNN,
                            self).get_config()  # get RNN's config
        del base_config['cell']
        cell_config = self.cell.get_config()
        return {**base_config, **cell_config}
