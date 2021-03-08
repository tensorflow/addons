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
# =============================================================================

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Addons")
class SpectralNorm(tf.keras.constraints.Constraint):
    """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the kernel weights by
    constraining its spectral norm, which can stabilize the training of GANs.

    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).

    Spectral normalization for `tf.keras.layers.Conv2D`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = tf.keras.layers.Conv2D(2, (2, 2), kernel_constraint=SpectralNorm(2))
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])

    Spectral normalization for `tf.keras.layers.Dense`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = tf.keras.layers.Dense(10, kernel_constraint=SpectralNorm(10))
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])

    Args:
      output_channels: `int`, the dimensionality of the output space of the kernel (i.e. the number of output filters).
      power_iterations: `int`, the number of iterations during normalization.
      dtype: The dtype of the layer's computations and weights. Can also be a `tf.keras.mixed_precision.Policy`,
        which allows the computation and weight dtype to differ. Default of `None` means to use
        `tf.keras.mixed_precision.global_policy()`.
    Raises:
      ValueError: If the product of the components of the kernel shape is not divisible by `output_channels`.
      InvalidArgumentError: When executing eagerly, if the product of the components of the kernel shape is not
        divisible by `output_channels`.
      ValueError: If initialized with negative `power_iterations`.
    """

    def __init__(self, output_channels, power_iterations=1, dtype=None):
        self.output_channels = output_channels
        self.power_iterations = power_iterations

        # set dtype & compute_dtype
        self._set_dtype_policy(dtype)

        self.u = tf.Variable(
            tf.random.truncated_normal(shape=(1, self.output_channels), stddev=0.02),
            trainable=False,
            dtype=self.dtype,
            name="sn_u",
        )

    def _set_dtype_policy(self, dtype):
        """Sets self._dtype_policy."""
        if isinstance(dtype, tf.keras.mixed_precision.Policy):
            self._dtype_policy = dtype
        elif isinstance(dtype, dict):
            self._dtype_policy = tf.keras.utils.deserialize_keras_object(dtype)
        elif dtype:
            self._dtype_policy = tf.keras.mixed_precision.Policy(
                tf.dtypes.as_dtype(dtype).name
            )
        else:
            self._dtype_policy = tf.keras.mixed_precision.global_policy()

    @property
    def dtype(self):
        """The dtype of the constraint weights."""
        return self._dtype_policy.variable_dtype

    @property
    def compute_dtype(self):
        """The dtype of the constraints' computations."""
        return self._dtype_policy.compute_dtype

    @tf.function
    def _normalize_weights(self, w):
        """Generate spectral normalized weights.

        This method will update the value of `self.w` with the
        spectral normalized value.
        """

        w_flat = tf.reshape(
            tf.cast(w, dtype=self.compute_dtype), [-1, self.output_channels]
        )
        u = tf.cast(self.u, dtype=self.compute_dtype)

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w_flat, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w_flat))

            sigma = tf.matmul(tf.matmul(v, w_flat), u, transpose_b=True)

            self.u.assign(tf.cast(u, dtype=self.dtype))

        return w / tf.cast(sigma, dtype=w.dtype)

    def __call__(self, w):
        return self._normalize_weights(w)

    def get_config(self):
        return {
            "output_channels": self.output_channels,
            "power_iterations": self.power_iterations,
            "dtype": self.dtype,
        }


class SpectralNormBuilder(object):
    """Factory for SpectralNorm constraints which tracks the variables in the constraints for easy checkpointing

    >>> spectral_norm = SpectralNormBuilder()
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.layers.Input(shape=[1]))
    >>> model.add(tf.keras.layers.Dense(1, kernel_constraint=spectral_norm.build(1)))
    >>> model.add(tf.keras.layers.Dense(2, kernel_constraint=spectral_norm.build(2)))
    >>> checkpoint = tf.train.Checkpoint(model=model, spectral_norm_variables=spectral_norm.variables)

    """

    def __init__(self):
        self._var_list = []

    @property
    def variables(self):
        """Returns list of all variables which the factory has built"""
        return self._var_list

    def build(self, output_channels, power_iterations=1, dtype=None):
        """Creates and returns a spectral normalization constraint

        Args:
          output_channels: `int`, the dimensionality of the output space of the kernel (i.e. the number of output filters).
          power_iterations: `int`, the number of iterations during normalization.
          dtype: The dtype of the layer's computations and weights. Can also be a `tf.keras.mixed_precision.Policy`,
            which allows the computation and weight dtype to differ. Default of `None` means to use
            `tf.keras.mixed_precision.global_policy()`.
        Returns:
          A new SpectralNorm object
        """
        new_constraint = SpectralNorm(output_channels, power_iterations, dtype)
        self._var_list.append(new_constraint.u)
        return new_constraint
