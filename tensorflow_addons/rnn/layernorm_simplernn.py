# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Module for LayernormSimpleRNN and LayernormSimpleRNNCell."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, SimpleRNNCell
from tensorflow.keras.layers import LayerNormalization

from tensorflow.python.keras import backend as K  # for SimpleRNNCell.call()
# from tensorflow.python.keras import regularizers  # for activity_regularizer
# from tensorflow.python.keras.engine.input_spec import InputSpec  # for SimpleRNN.__init__()
# from tensorflow.python.keras.utils import tf_utils  # for shape_type_conversion


@tf.keras.utils.register_keras_serializable(package='Addons')
# class LayernormSimpleRNNCell(SimpleRNNCell, LayerNormalization):
class LayernormSimpleRNNCell(SimpleRNNCell):
    """Cell class for LayernormSimpleRNN.

    Motivation:
    - Drop-In Replacement for keras.layers.SimpleRNNCell
    - demonstrate how to add LayerNormalization to all RNNs as option
    - see Ba et al. (2016), and tf.keras.layers.LayerNormalization

    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.

    This class processes one step within the whole time sequence input, whereas
    `tf.keras.layer.LayernormSimpleRNN` processes the whole sequence.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      use_layernorm: Boolean, (default `False`), whether layer uses layer
        normalization instead of a bias vector.
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
      recurrent_constraint: Constraint function applied to the `recurrent_kernel`
        weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_constraint: Constraint function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
        the linear transformation of the recurrent state. Default: 0.

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
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    rnn = tf.keras.layers.RNN(
      tf.keras.layers.LayernormSimpleRNNCell(4, use_layernorm=True))

    output = rnn(inputs)  # The output has shape `[32, 4]`.

    rnn = tf.keras.layers.RNN(
        tf.keras.layers.LayernormSimpleRNNCell(4, use_layernorm=True),
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
            use_layernorm=False,  # NEW(!)
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
        SimpleRNNCell.__init__(
            self,
            units,
            activation=activation,
            use_bias=False if use_layernorm else use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=None if use_layernorm else bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=None if use_layernorm else bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=None if use_layernorm else bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        if use_layernorm:
            # LayerNormalization.__init__(self,
            self.layernorm = LayerNormalization(
                axis=-1,
                epsilon=layernorm_epsilon,
                center=True,
                scale=True,
                beta_initializer=bias_initializer,
                gamma_initializer=gamma_initializer,
                beta_regularizer=bias_regularizer,
                gamma_regularizer=gamma_regularizer,
                beta_constraint=bias_constraint,
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
        prev_output = states[0]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + K.dot(prev_output, self.recurrent_kernel)

        if self.use_layernorm:
            # output = LayerNormalization.call(self, output)
            output = self.layernorm(output)

        if self.activation is not None:
            output = self.activation(output)

        return output, [output]

    # use SimpleRNNCell's get_initial_state method

    def get_config(self):
        config = {'use_layernorm': self.use_layernorm}
        cell_config = SimpleRNNCell.get_config(self)
        del cell_config['name']
        if self.use_layernorm:
            # ln_config = LayerNormalization.get_config(self)
            ln_config = self.layernorm.get_config()
            ln_config['bias_initializer'] = ln_config.pop("beta_initializer")
            ln_config['bias_regularizer'] = ln_config.pop("beta_regularizer")
            ln_config['bias_constraint'] = ln_config.pop("beta_constraint")
            ln_config['layernorm_epsilon'] = ln_config.pop("epsilon")
            del ln_config['axis']
            del ln_config['center']
            del ln_config['scale']
            del ln_config['name']
        else:
            ln_config = {}
        return {**config, **cell_config, **ln_config}


@tf.keras.utils.register_keras_serializable(package='Addons')
class LayernormSimpleRNN(SimpleRNN):
    """Fully-connected RNN where the output is to be fed back to input.

    Motivation:
    - Drop-In Replacement for keras.layers.SimpleRNN
    - demonstrate how to add LayerNormalization to all RNNs as option
    - see Ba et al. (2016), and tf.keras.layers.LayerNormalization

    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      use_layernorm: Boolean, (default `False`), whether layer uses layer
        normalization instead of a bias vector.
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
      recurrent_constraint: Constraint function applied to the `recurrent_kernel`
        weights matrix.  Default: `None`.
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
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    model = tf.keras.layers.LayernormSimpleRNN(4, use_layernorm=True)

    output = model(inputs)  # The output has shape `[32, 4]`.

    model = tf.keras.layers.LayernormSimpleRNN(
        4, use_layernorm=True, return_sequences=True, return_state=True)

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
            use_layernorm=False,  # NEW(!)
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
        super(SimpleRNN, self).__init__(  # call RNN's init
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
        base_config = super(SimpleRNN, self).get_config()  # get RNN's config
        del base_config['cell']
        cell_config = self.cell.get_config()
        return {**base_config, **cell_config}
