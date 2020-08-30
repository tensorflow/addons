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


@tf.keras.utils.register_keras_serializable(package="Addons")
class NoisyDense(tf.keras.layers.Layer):
    """
    Like normal dense layer (https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/layers/core.py#L1067-L1233)
    but random noisy is added to the weights matrix. But as the network improves the random noise is decayed until it is insignificant.

    A `NoisyDense` layer implements the operation:
    `output = activation(dot(input, µ_kernel + (σ_kernel * ε_kernel)) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `µ_kernel` is your average weights matrix
    created by the layer, σ_kernel is a weights matrix that controls the importance of
    the ε_kernel which is just random noise, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Example:
    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
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

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(NoisyDense, self).__init__(
            activity_regularizer=activity_regularizer, **kwargs
        )

        self.units = int(units) if not isinstance(units, int) else units
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

        self.σ_init = initializers.Constant(value=0.5 / sqrt_dim)
        self.µ_init = initializers.RandomUniform(
            minval=-1 / sqrt_dim, maxval=1 / sqrt_dim
        )

        # Learnable parameters
        # Agent will learn to decay σ as it improves creating a sort of learned epsilon decay
        self.σ_kernel = self.add_weight(
            "σ_kernel",
            shape=[self.last_dim, self.units],
            initializer=self.σ_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        self.µ_kernel = self.add_weight(
            "µ_kernel",
            shape=[self.last_dim, self.units],
            initializer=self.µ_init,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.σ_bias = self.add_weight(
                "σ_bias",
                shape=[self.units,],
                initializer=self.σ_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

            self.µ_bias = self.add_weight(
                "µ_bias",
                shape=[self.units,],
                initializer=self.µ_init,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )

        self.built = True

    @staticmethod
    def _scale_noise(x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def call(self, inputs):
        dtype = self._compute_dtype_object
        if inputs.dtype.base_dtype != dtype.base_dtype:
            inputs = tf.cast(inputs, dtype=dtype)

        # Fixed parameters added as the noise
        ε_i = tf.random.normal([self.last_dim, self.units], dtype=dtype)
        ε_j = tf.random.normal([self.units,], dtype=dtype)

        # Creates the factorised Gaussian noise
        f = NoisyDense._scale_noise
        ε_kernel = f(ε_i) * f(ε_j)
        ε_bias = f(ε_j)

        # Performs: y = (µw + σw · εw)x + µb + σb · εb
        # to calculate the output
        kernel = self.µ_kernel + (self.σ_kernel * ε_kernel)

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
            noisy_bias = self.µ_bias + (self.σ_bias * ε_bias)
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
