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
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs.

    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).

    >>> x = [ 177  164  169  165  133  134  168  174  160  165; 154  146  100   54   48   19    8   6
    ... 4  148  132; 162  152   94  200  177   73   27    1  145  153; 186   56  177  202  176  132   51
    ... 13   43  152; 186   17  151  162  119  106   70   11   25  100; 146   12  154  141  140  146
    ... 92   21   16  101;  93    9  192  150  148  148   42   26   19   92;  91   13  125  129  114
    ... 127   42   27    6   86; 113   43    9  190  125   58   38   26   12   99; 134   29   19   17
    ... 72   42   61   34   12   83; 123   26   36   42  154  118  138   23   29   44;  74   34   76  18
    ... 7  186  183  181   73   50   54;  70   27  194  204  195  193  174   58   39   96]
    >>> x = uint8(x)
    >>> net = SpectralNormalization(
    ... tf.keras.layers.Conv2D(2, 2, activation="relu"),
    ... input_shape=(32, 32, 3))(x)
    >>> net = SpectralNormalization(
    ... tf.keras.layers.Conv2D(16, 5, activation="relu"))(net)
    >>> net = SpectralNormalization(
    ... tf.keras.layers.Dense(120, activation="relu"))(net)
    >>> net = SpectralNormalization(
    ... tf.keras.layers.Dense(n_classes))(net)

    Arguments:
      layer: A `tf.keras.layers.Layer` instance that
        has either `kernel` or `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.
    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings` attribute.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

            self.w.assign(self.w / sigma)
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}
