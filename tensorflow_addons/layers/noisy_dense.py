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


def _scale_noise(x):
    return tf.sign(x) * tf.sqrt(tf.abs(x))


@tf.keras.utils.register_keras_serializable(package="Addons")
class NoisyDense(tf.keras.layers.Layer):
    r"""Like normal dense layer (https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/layers/core.py#L1067-L1233)
    but random noise is added to the weights matrix. As the network improves the random noise is decayed until it is insignificant.

    A `NoisyDense` layer implements the operation:
    $$
    \mathrm{NoisyDense}(x) = \mathrm{activation}(\mathrm{dot}(x, \mu + (\sigma \cdot \epsilon)) + \mathrm{bias})
    $$
    with bias only being added if `use_bias` is `True`.

    Example:
    >>> # Create a `Sequential` model and add a NoisyDense layer as the first layer.
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

    Arguments:
    units: Positive integer, dimensionality of the output space.
    sigma: A float between 0-1 used as a standard deviation figure and is
      applied to the gaussian noise layer (`sigma_kernel` and `sigma_bias`).
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
    """

    @typechecked
    def __init__(
        self,
        units: int,
        sigma: float = 0.5,
        activation: types.Activation = None,
        use_bias: bool = True,
        kernel_regularizer: types.Regularizer = None,
        bias_regularizer: types.Regularizer = None,
        activity_regularizer: types.Regularizer = None,
        kernel_constraint: types.Constraint = None,
        bias_constraint: types.Constraint = None,
        **kwargs
    ):
        super(NoisyDense, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )

        self.units = units
        self.sigma = sigma
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

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

        self.sigma_init = initializers.Constant(value=self.sigma / sqrt_dim)
        self.mu_init = initializers.RandomUniform(
            minval=-1 / sqrt_dim, maxval=1 / sqrt_dim
        )

        # Learnable parameters
        # Agent will learn to decay sigma as it improves creating a sort of learned epsilon decay
        self.sigma_kernel = self.add_weight(
            "sigma_kernel",
            shape=[self.last_dim, self.units],
            initializer=self.sigma_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.mu_kernel = self.add_weight(
            "mu_kernel",
            shape=[self.last_dim, self.units],
            initializer=self.mu_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.sigma_bias = self.add_weight(
                "sigma_bias",
                shape=[
                    self.units,
                ],
                initializer=self.sigma_init,
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
                initializer=self.mu_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.sigma_bias = None
            self.mu_bias = None
        self.built = True

    # Create the factorised Gaussian noise
    def reset_noise(self):
        dtype = self._compute_dtype_object

        # Generate random noise
        ε_i = tf.random.normal([self.last_dim, self.units], dtype=dtype)
        ε_j = tf.random.normal(
            [
                self.units,
            ],
            dtype=dtype,
        )

        # Scale the random noise
        self.ε_kernel = _scale_noise(ε_i) * _scale_noise(ε_j)
        self.ε_bias = _scale_noise(ε_j)

    def remove_noise(self):
        dtype = self._compute_dtype_object
        self.ε_kernel = tf.zeros([self.last_dim, self.units], dtype=dtype)
        self.ε_bias = tf.zeros([self.last_dim, self.units], dtype=dtype)

    def call(self, inputs, reset_noise=True):
        dtype = self._compute_dtype_object
        if inputs.dtype.base_dtype != dtype.base_dtype:
            inputs = tf.cast(inputs, dtype=dtype)

        # Generate fixed parameters added as the noise
        if reset_noise:
            self.reset_noise()

        # Performs: y = (muw + sigmaw · εw)x + mub + sigmab · εb
        # to calculate the output
        kernel = self.mu_kernel + (self.sigma_kernel * self.ε_kernel)

        if inputs.dtype.base_dtype != dtype.base_dtype:
            inputs = tf.cast(inputs, dtype=dtype)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            if isinstance(inputs, tf.sparse.SparseTensor):
                outputs = tf.sparse.sparse_dense_matmul(inputs, kernel)
            else:
                outputs = tf.linalg.matmul(inputs, kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            noisy_bias = self.mu_bias + (self.sigma_bias * self.ε_bias)
            outputs = tf.nn.bias_add(outputs, noisy_bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The innermost dimension of input_shape must be defined, but saw: %s"
                % input_shape
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super(NoisyDense, self).get_config()
        config.update(
            {
                "units": self.units,
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
