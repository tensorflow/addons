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
"""Implements RMC Cell."""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow_addons.utils.types import (
    Activation,
    Initializer,
    Regularizer,
    Constraint,
)

from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class RMCCell(keras.layers.AbstractRNNCell):
    """Relational Memory Core (RMC) cell.

    This implements the recurrent cell from the paper:

        https://papers.nips.cc/paper/2018/file/e2eabaf96372e20a9e3d4b5f83723a61-Paper.pdf

    A. Santoro, et al.
    "Relational recurrent neural networks".
    Advances in neural information processing systems, 2018.

    Example:

    >>> inputs = np.random.random([30,23,9]).astype(np.float32)
    >>> RMCCell = tfa.rnn.RMCCell(2, 2, 1)
    >>> rnn = tf.keras.layers.RNN(RMCCell, return_sequences=True, return_state=True)
    >>> outputs, memory_state = rnn(inputs, initial_state=RMCCell.get_initial_state(30))
    >>> outputs.shape
    TensorShape([30, 23, 4])
    >>> memory_state.shape
    TensorShape([30, 4])

    """

    @typechecked
    def __init__(
        self,
        n_slots: int,
        n_heads: int,
        head_size: int,
        n_blocks: int = 1,
        n_layers: int = 3,
        activation: Activation = "tanh",
        recurrent_activation: Activation = "hard_sigmoid",
        forget_bias: float = 1.0,
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "orthogonal",
        bias_initializer: Initializer = "zeros",
        kernel_regularizer: Regularizer = None,
        recurrent_regularizer: Regularizer = None,
        bias_regularizer: Regularizer = None,
        kernel_constraint: Constraint = None,
        recurrent_constraint: Constraint = None,
        bias_constraint: Constraint = None,
        **kwargs,
    ):
        """Initializes the parameters for a RMC cell.

        Args:
            n_slots: Number of memory slots.
            n_heads: Number of attention heads.
            head_size: Size of each attention head.
            n_blocks: Number of feed-forward networks.
            n_layers: Amout of layers per feed-forward network.
            activation: Output activation function.
            recurrent_activation: Recurrent step activation function.
            forget_bias: Forget gate bias values.
            kernel_initializer: Kernel initializer function.
            recurrent_initializer: Recurrent kernel initializer function.
            bias_initializer: Bias initializer function.
            kernel_regularizer: Kernel regularizer function.
            recurrent_regularizer: Recurrent kernel regularizer function.
            bias_regularizer: Bias regularizer function.
            kernel_constraint: Kernel constraint function.
            recurrent_constraint: Recurrent kernel constraint function.
            bias_constraint: Bias constraint function.

        """
        super().__init__(**kwargs)
        self.n_slots = n_slots
        self.slot_size = n_heads * head_size
        self.n_heads = n_heads
        self.head_size = head_size
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.units = self.slot_size * n_slots
        self.n_gates = 2 * self.slot_size

        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.forget_bias = forget_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.projector = keras.layers.Dense(self.slot_size)
        self.before_norm = keras.layers.LayerNormalization()
        self.linear = [
            keras.layers.Dense(self.slot_size, activation="relu")
            for _ in range(n_layers)
        ]
        self.after_norm = keras.layers.LayerNormalization()
        self.attn = keras.layers.MultiHeadAttention(self.slot_size, self.n_heads)

    @property
    def state_size(self):
        return [self.units, self.units]

    @property
    def output_size(self):
        return self.units

    def build(self, inputs_shape):
        # Variables for the RMC cell. `kernel` stands for the `W` matrix,
        # `recurrent_kernel` stands for the `U` matrix and
        # `bias` stands for the `b` array
        self.kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name="kernel",
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.slot_size, self.n_gates),
            name="recurrent_kernel",
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
        )
        self.bias = self.add_weight(
            shape=(self.n_gates,),
            name="bias",
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
        )

        self.built = True

    def _attend_over_memory(self, inputs, memory):
        for _ in range(self.n_blocks):
            concat_memory = tf.concat([inputs, memory], 1)
            att_memory, _ = self.attn(memory, concat_memory, concat_memory)
            norm_memory = self.before_norm(att_memory + memory)
            linear_memory = norm_memory

            for l in self.linear:
                linear_memory = l(linear_memory)

            memory = self.after_norm(norm_memory + linear_memory)

        return memory

    def call(self, inputs, states):
        # Gathers previous hidden and memory states
        h_prev, m_prev = states

        # Projects the inputs to the same size as the memory
        inputs = tf.expand_dims(self.projector(inputs), 1)

        # Reshapes the previous hidden state and memory tensors
        h_prev = tf.reshape(h_prev, [h_prev.shape[0], self.n_slots, self.slot_size])
        m_prev = tf.reshape(m_prev, [m_prev.shape[0], self.n_slots, self.slot_size])

        # Copies the inputs for the forget and input gates
        inputs_f = inputs
        inputs_i = inputs

        # Splits up the kernel into forget and input gates kernels
        # Also calculates the forget and input gates kernel outputs
        k_f, k_i = tf.split(self.kernel, 2, axis=1)
        x_f = tf.tensordot(inputs_f, k_f, axes=[[-1], [0]])
        x_i = tf.tensordot(inputs_i, k_i, axes=[[-1], [0]])

        # Splits up the recurrent kernel into forget and input gates kernels
        # Also calculates the forget and input gates recurrent kernel outputs
        rk_f, rk_i = tf.split(self.recurrent_kernel, 2, axis=1)
        x_f += tf.tensordot(h_prev, rk_f, axes=[[-1], [0]])
        x_i += tf.tensordot(h_prev, rk_i, axes=[[-1], [0]])

        # Splits up the bias into forget and input gates biases
        # Also adds the forget and input gates bias outputs
        b_f, b_i = tf.split(self.bias, 2, axis=0)
        x_f = tf.nn.bias_add(x_f, b_f)
        x_i = tf.nn.bias_add(x_i, b_i)

        # Calculates the attention mechanism over the previous memory
        # Also calculates current memory and hidden states
        att_m = self._attend_over_memory(inputs, m_prev)
        m = self.recurrent_activation(
            x_f + self.forget_bias
        ) * m_prev + self.recurrent_activation(x_i) * self.activation(att_m)
        h = self.activation(m)

        # Reshapes both current hidden and memory states to their correct output size
        h = tf.reshape(h, [h.shape[0], self.units])
        m = tf.reshape(m, [m.shape[0], self.units])

        return h, [h, m]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        states = tf.eye(self.n_slots, batch_shape=[batch_size], dtype=dtype)

        if self.slot_size > self.n_slots:
            diff = self.slot_size - self.n_slots
            padding = tf.zeros((batch_size, self.n_slots, diff))
            states = tf.concat([states, padding], -1)

        elif self.slot_size < self.n_slots:
            states = states[:, :, : self.slot_size]

        states = tf.reshape(states, (states.shape[0], -1))

        return states, states

    def get_config(self):
        config = {
            "n_slots": self.n_slots,
            "n_heads": self.n_heads,
            "head_size": self.head_size,
            "n_blocks": self.n_blocks,
            "n_layers": self.n_layers,
            "activation": tf.keras.activations.serialize(self.activation),
            "recurrent_activation": tf.keras.activations.serialize(
                self.recurrent_activation
            ),
            "forget_bias": self.forget_bias,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "recurrent_initializer": tf.keras.initializers.serialize(
                self.recurrent_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "recurrent_regularizer": tf.keras.regularizers.serialize(
                self.recurrent_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
            "recurrent_constraint": tf.keras.constraints.serialize(
                self.recurrent_constraint
            ),
            "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}
