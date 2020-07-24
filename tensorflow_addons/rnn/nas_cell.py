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
"""Implements NAS Cell."""

import tensorflow as tf
import tensorflow.keras as keras
from typeguard import typechecked

from tensorflow_addons.utils.types import (
    FloatTensorLike,
    TensorLike,
    Initializer,
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
