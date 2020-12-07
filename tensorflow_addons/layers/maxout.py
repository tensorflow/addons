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
"""Implementing Maxout layer."""

import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class Maxout(tf.keras.layers.Layer):
    """Applies Maxout to the input.

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville, Yoshua Bengio. https://arxiv.org/abs/1302.4389

    Usually the operation is performed in the filter/channel dimension. This
    can also be used after Dense layers to reduce number of features.

    Args:
      num_units: Specifies how many features will remain after maxout
        in the `axis` dimension (usually channel).
        This must be a factor of number of features.
      axis: The dimension where max pooling will be performed. Default is the
        last dimension.

    Input shape:
      nD tensor with shape: `(batch_size, ..., axis_dim, ...)`.

    Output shape:
      nD tensor with shape: `(batch_size, ..., num_units, ...)`.
    """

    @typechecked
    def __init__(self, num_units: int, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        self.axis = axis

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = tf.shape(inputs)[i]

        num_channels = shape[self.axis]
        if not isinstance(num_channels, tf.Tensor) and num_channels % self.num_units:
            raise ValueError(
                "number of features({}) is not "
                "a multiple of num_units({})".format(num_channels, self.num_units)
            )

        if self.axis < 0:
            axis = self.axis + len(shape)
        else:
            axis = self.axis
        assert axis >= 0, "Find invalid axis: {}".format(self.axis)

        expand_shape = shape[:]
        expand_shape[axis] = self.num_units
        k = num_channels // self.num_units
        expand_shape.insert(axis, k)

        outputs = tf.math.reduce_max(
            tf.reshape(inputs, expand_shape), axis, keepdims=False
        )
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[self.axis] = self.num_units
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = {"num_units": self.num_units, "axis": self.axis}
        base_config = super().get_config()
        return {**base_config, **config}
