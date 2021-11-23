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

import tensorflow as tf
from tensorflow.keras import (
    activations,
    initializers,
    regularizers,
    constraints,
)
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from typeguard import typechecked

from tensorflow_addons.utils import types


def _scaled_noise(size, dtype):
    x = tf.random.normal(shape=size, dtype=dtype)
    return tf.sign(x) * tf.sqrt(tf.abs(x))


@tf.keras.utils.register_keras_serializable(package="Addons")
class NoisyDense(tf.keras.layers.Dense):
    r"""Noisy dense layer that injects random noise to the weights of dense layer.

    Noisy dense layers are fully connected layers whose weights and biases are
    augmented by factorised Gaussian noise. The factorised Gaussian noise is
    controlled through gradient descent by a second weights layer.

    A `NoisyDense` layer implements the operation:
    $$
    \mathrm{NoisyDense}(x) =
    \mathrm{activation}(\mathrm{dot}(x, \mu + (\sigma \cdot \epsilon))
    + \mathrm{bias})
    $$
    where $\mu$ is the standard weights layer, $\epsilon$ is the factorised
    Gaussian noise, and $\sigma$ is a second weights layer which controls
    $\epsilon$.

    Note: bias only added if `use_bias` is `True`.

    Example:

    >>> # Create a `Sequential` model and add a NoisyDense
    >>> # layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(NoisyDense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(NoisyDense(32))
    >>> model.output_shape
    (None, 32)

    There are implemented both variants:
        1. Independent Gaussian noise
        2. Factorised Gaussian noise.
    We can choose between that by 'use_factorised' parameter.

    Args:
      units: Positive integer, dimensionality of the output space.
      sigma: A float between 0-1 used as a standard deviation figure and is
        applied to the gaussian noise layer (`sigma_kernel` and `sigma_bias`). (uses only if use_factorised=True)
      use_factorised: Boolean, whether the layer uses independent or factorised Gaussian noise
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation").
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.

    References:
      - [Noisy Networks for Explanation](https://arxiv.org/pdf/1706.10295.pdf)
    """

    @typechecked
    def __init__(
        self,
        units: int,
        sigma: float = 0.5,
        use_factorised: bool = True,
        activation: types.Activation = None,
        use_bias: bool = True,
        kernel_regularizer: types.Regularizer = None,
        bias_regularizer: types.Regularizer = None,
        activity_regularizer: types.Regularizer = None,
        kernel_constraint: types.Constraint = None,
        bias_constraint: types.Constraint = None,
        **kwargs,
    ):
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        delattr(self, "kernel_initializer")
        delattr(self, "bias_initializer")
        self.sigma = sigma
        self.use_factorised = use_factorised

    def build(self, input_shape):
        # Make sure dtype is correct
        dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense` layer with non-floating point "
                "dtype %s" % (dtype,)
            )

        input_shape = tf.TensorShape(input_shape)
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        sqrt_dim = self.last_dim ** (1 / 2)
        if self.last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.last_dim})

        # use factorising Gaussian variables
        if self.use_factorised:
            mu_init = 1.0 / sqrt_dim
            sigma_init = self.sigma / sqrt_dim
        # use independent Gaussian variables
        else:
            mu_init = (3.0 / self.last_dim) ** (1 / 2)
            sigma_init = 0.017

        sigma_init = initializers.Constant(value=sigma_init)
        mu_init = initializers.RandomUniform(minval=-mu_init, maxval=mu_init)

        # Learnable parameters
        self.sigma_kernel = self.add_weight(
            "sigma_kernel",
            shape=[self.last_dim, self.units],
            initializer=sigma_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.mu_kernel = self.add_weight(
            "mu_kernel",
            shape=[self.last_dim, self.units],
            initializer=mu_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.eps_kernel = self.add_weight(
            "eps_kernel",
            shape=[self.last_dim, self.units],
            initializer=initializers.Zeros(),
            regularizer=None,
            constraint=None,
            dtype=self.dtype,
            trainable=False,
        )

        if self.use_bias:
            self.sigma_bias = self.add_weight(
                "sigma_bias",
                shape=[
                    self.units,
                ],
                initializer=sigma_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.mu_bias = self.add_weight(
                "mu_bias",
                shape=[
                    self.units,
                ],
                initializer=mu_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.eps_bias = self.add_weight(
                "eps_bias",
                shape=[
                    self.units,
                ],
                initializer=initializers.Zeros(),
                regularizer=None,
                constraint=None,
                dtype=self.dtype,
                trainable=False,
            )
        else:
            self.sigma_bias = None
            self.mu_bias = None
            self.eps_bias = None
        self.reset_noise()
        self.built = True

    @property
    def kernel(self):
        return self.mu_kernel + (self.sigma_kernel * self.eps_kernel)

    @property
    def bias(self):
        if self.use_bias:
            return self.mu_bias + (self.sigma_bias * self.eps_bias)

    def reset_noise(self):
        """Create the factorised Gaussian noise."""

        if self.use_factorised:
            # Generate random noise
            in_eps = _scaled_noise([self.last_dim, 1], dtype=self.dtype)
            out_eps = _scaled_noise([1, self.units], dtype=self.dtype)

            # Scale the random noise
            self.eps_kernel.assign(tf.matmul(in_eps, out_eps))
            self.eps_bias.assign(out_eps[0])
        else:
            # generate independent variables
            self.eps_kernel.assign(
                tf.random.normal(shape=[self.last_dim, self.units], dtype=self.dtype)
            )
            self.eps_bias.assign(
                tf.random.normal(
                    shape=[
                        self.units,
                    ],
                    dtype=self.dtype,
                )
            )

    def remove_noise(self):
        """Remove the factorised Gaussian noise."""

        self.eps_kernel.assign(tf.zeros([self.last_dim, self.units], dtype=self.dtype))
        self.eps_bias.assign(tf.zeros([self.units], dtype=self.dtype))

    def call(self, inputs):
        # TODO(WindQAQ): Replace this with `dense()` once public.
        return super().call(inputs)

    def get_config(self):
        # TODO(WindQAQ): Get rid of this hacky way.
        config = super(tf.keras.layers.Dense, self).get_config()
        config.update(
            {
                "units": self.units,
                "sigma": self.sigma,
                "use_factorised": self.use_factorised,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config
