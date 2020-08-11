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
"""Implements Polynomial Crossing Layer."""

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
class PolynomialCrossing(tf.keras.layers.Layer):
    """Layer for Deep & Cross Network to learn explicit feature interactions.

    A layer that applies feature crossing in learning certain explicit
    bounded-degree feature interactions more efficiently. The `call` method
    accepts `inputs` as a tuple of size 2 tensors. The first input `x0` should be
    the input to the first `PolynomialCrossing` layer in the stack, or the input
    to the network (usually after the embedding layer), the second input `xi`
    is the output of the previous `PolynomialCrossing` layer in the stack, i.e.,
    the i-th `PolynomialCrossing` layer.

    The output is `x[i+1] = x0 .* (W * x[i] + diag_scale * x[i]) + bias + x[i]`, where .* designates elementwise
    multiplication, W could be a full rank matrix, or a low rank matrix `U*V` to reduce the computational cost,
    and diag_scale increases the diagonal of W to improve training stability (especially for the low rank case).

    See [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf).

    Example:

    ```python
    # after embedding layer in a functional model:
    input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
    x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6))
    x1 = PolynomialCrossing(projection_dim=None)((x0, x0))
    x2 = PolynomialCrossing(projection_dim=None)((x0, x1))
    logits = tf.keras.layers.Dense(units=10)(x2)
    model = tf.keras.Model(input, logits)
    ```

    Arguments:
        projection_dim: project dimension to reduce the computational cost.
          Default is `None` such that a full (`input_dim` by `input_dim`)
          matrix W is used. If enabled, a low-rank matrix W = U*V will be used,
          where U is of size `input_dim` by `projection_dim` and V is of size
          `projection_dim` by `input_dim`. `projection_dim` need to be smaller
          than `input_dim`/2 to improve the model efficiency.
        diag_scale: a non-negative float used to increase the diagonal of the
           kernel W by `diag_scale`.
        use_bias: whether to calculate the bias/intercept for this layer. If set to
          False, no bias/intercept will be used in calculations, e.g., the data is
          already centered.
        kernel_initializer: Initializer instance to use on the kernel matrix.
        bias_initializer: Initializer instance to use on the bias vector.
        kernel_regularizer: Regularizer instance to use on the kernel matrix.
        bias_regularizer: Regularizer instance to use on bias vector.

    Input shape:
        A tuple of 2 `(batch_size, input_dim)` dimensional inputs.

    Output shape:
        A single `(batch_size, input_dim)` dimensional output.
    """

    @typechecked
    def __init__(
        self,
        projection_dim: int = None,
        diag_scale: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: types.Initializer = "truncated_normal",
        bias_initializer: types.Initializer = "zeros",
        kernel_regularizer: types.Regularizer = None,
        bias_regularizer: types.Regularizer = None,
        **kwargs
    ):
        super(PolynomialCrossing, self).__init__(**kwargs)

        self.projection_dim = projection_dim
        self.diag_scale = diag_scale
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                "Input shapes must be a tuple or list of size 2, "
                "got {}".format(input_shape)
            )
        last_dim = input_shape[-1][-1]
        if self.projection_dim is None:
            self.kernel = self.add_weight(
                "kernel",
                shape=[last_dim, last_dim],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            if self.projection_dim < 0 or self.projection_dim > last_dim / 2:
                raise ValueError(
                    "`projection_dim` should be smaller than last_dim / 2 to improve"
                    "the model efficiency, and should be positive. Got "
                    "`projection_dim` {}, and last dimension of input {}".format(
                        self.projection_dim, last_dim
                    )
                )
            self.kernel_u = self.add_weight(
                "kernel_u",
                shape=[last_dim, self.projection_dim],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                dtype=self.dtype,
                trainable=True,
            )
            self.kernel_v = self.add_weight(
                "kernel_v",
                shape=[self.projection_dim, last_dim],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                dtype=self.dtype,
                trainable=True,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[last_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                dtype=self.dtype,
                trainable=True,
            )
        self.built = True

    def call(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(
                "Inputs to the layer must be a tuple or list of size 2, "
                "got {}".format(inputs)
            )
        x0, x = inputs
        if self.projection_dim is None:
            prod_output = tf.matmul(x, self.kernel)
        else:
            prod_output = tf.matmul(x, self.kernel_u)
            prod_output = tf.matmul(prod_output, self.kernel_v)
        if self.diag_scale:
            prod_output = tf.add(prod_output, self.diag_scale * x)
        outputs = x0 * prod_output + x
        if self.use_bias:
            outputs = tf.add(outputs, self.bias)
        return outputs

    def get_config(self):
        config = {
            "projection_dim": self.projection_dim,
            "diag_scale": self.diag_scale,
            "use_bias": self.use_bias,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
        }
        base_config = super(PolynomialCrossing, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)):
            raise ValueError(
                "A `PolynomialCrossing` layer should be called " "on a list of inputs."
            )
        return input_shape[0]
