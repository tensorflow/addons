# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implements Echo State recurrent Network (ESN) layer."""

import tensorflow as tf
from tensorflow_addons.rnn import ESNCell
from typeguard import typechecked

from tensorflow_addons.utils.types import (
    Activation,
    FloatTensorLike,
    TensorLike,
    Initializer,
)


@tf.keras.utils.register_keras_serializable(package="Addons")
class ESN(tf.keras.layers.RNN):
    """Echo State Network layer.

    This implements the recurrent layer using the ESNCell.

    This is based on the paper
        H. Jaeger
        ["The "echo state" approach to analysing and training recurrent neural networks"]
        (https://www.researchgate.net/publication/215385037).
        GMD Report148, German National Research Center for Information Technology, 2001.

    Arguments:
        units: Positive integer, dimensionality of the reservoir.
        connectivity: Float between 0 and 1.
            Connection probability between two reservoir units.
            Default: 0.1.
        leaky: Float between 0 and 1.
            Leaking rate of the reservoir.
            If you pass 1, it's the special case the model does not have leaky integration.
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
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            Default: `glorot_uniform`.
        recurrent_initializer: Initializer for the `recurrent_kernel` weights matrix,
            used for the linear transformation of the recurrent state.
            Default: `glorot_uniform`.
        bias_initializer: Initializer for the bias vector.
            Default: `zeros`.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.

    Call arguments:
        inputs: A 3D tensor.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the cell
            when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell.
    """

    @typechecked
    def __init__(
        self,
        units: TensorLike,
        connectivity: FloatTensorLike = 0.1,
        leaky: FloatTensorLike = 1,
        spectral_radius: FloatTensorLike = 0.9,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation: Activation = "tanh",
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "glorot_uniform",
        bias_initializer: Initializer = "zeros",
        return_sequences=False,
        go_backwards=False,
        unroll=False,
        **kwargs
    ):
        cell = ESNCell(
            units,
            connectivity=connectivity,
            leaky=leaky,
            spectral_radius=spectral_radius,
            use_norm2=use_norm2,
            use_bias=use_bias,
            activation=activation,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            dtype=kwargs.get("dtype"),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            go_backwards=go_backwards,
            unroll=unroll,
            **kwargs,
        )

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super().call(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
            constants=None,
        )

    @property
    def units(self):
        return self.cell.units

    @property
    def connectivity(self):
        return self.cell.connectivity

    @property
    def leaky(self):
        return self.cell.leaky

    @property
    def spectral_radius(self):
        return self.cell.spectral_radius

    @property
    def use_norm2(self):
        return self.cell.use_norm2

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def activation(self):
        return self.cell.activation

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

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
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
