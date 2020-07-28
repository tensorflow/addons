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
"""Implements PeepholeLSTM Cell."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Addons")
class PeepholeLSTMCell(tf.keras.layers.LSTMCell):
    """Equivalent to `tf.keras.layers.LSTMCell` class but adds peephole connections.

    Peephole connections allow the gates to utilize the previous internal state as
    well as the previous hidden state (which is what LSTMCell is limited to).
    This allows PeepholeLSTMCell to better learn precise timings over LSTMCell.

    From [Gers et al., 2002](
    http://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf):

    "We find that LSTM augmented by 'peephole connections' from its internal
    cells to its multiplicative gates can learn the fine distinction between
    sequences of spikes spaced either 50 or 49 time steps apart without the help
    of any short training exemplars."

    The peephole implementation is based on:

    [Sak et al., 2014](https://research.google.com/pubs/archive/43905.pdf)

    Example:

    ```python
    # Create 2 PeepholeLSTMCells
    peephole_lstm_cells = [PeepholeLSTMCell(size) for size in [128, 256]]
    # Create a layer composed sequentially of the peephole LSTM cells.
    layer = RNN(peephole_lstm_cells)
    input = keras.Input((timesteps, input_dim))
    output = layer(input)
    ```
    """

    def build(self, input_shape):
        super().build(input_shape)
        # The following are the weight matrices for the peephole connections. These
        # are multiplied with the previous internal state during the computation of
        # carry and output.
        self.input_gate_peephole_weights = self.add_weight(
            shape=(self.units,),
            name="input_gate_peephole_weights",
            initializer=self.kernel_initializer,
        )
        self.forget_gate_peephole_weights = self.add_weight(
            shape=(self.units,),
            name="forget_gate_peephole_weights",
            initializer=self.kernel_initializer,
        )
        self.output_gate_peephole_weights = self.add_weight(
            shape=(self.units,),
            name="output_gate_peephole_weights",
            initializer=self.kernel_initializer,
        )

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i
            + tf.keras.backend.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
            + self.input_gate_peephole_weights * c_tm1
        )
        f = self.recurrent_activation(
            x_f
            + tf.keras.backend.dot(
                h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2]
            )
            + self.forget_gate_peephole_weights * c_tm1
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + tf.keras.backend.dot(
                h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3]
            )
        )
        o = self.recurrent_activation(
            x_o
            + tf.keras.backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
            + self.output_gate_peephole_weights * c
        )
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0 + self.input_gate_peephole_weights * c_tm1)
        f = self.recurrent_activation(z1 + self.forget_gate_peephole_weights * c_tm1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
        return c, o
