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
"""Spatial Pyramid Pooling layers"""

import tensorflow as tf
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D
import tensorflow_addons.utils.keras_utils as conv_utils

from typeguard import typechecked
from typing import Union, Iterable


@tf.keras.utils.register_keras_serializable(package="Addons")
class SpatialPyramidPooling2D(tf.keras.layers.Layer):
    """Performs Spatial Pyramid Pooling.

    Original Paper: https://arxiv.org/pdf/1406.4729.pdf

    Arguments:
      bins: Either a collection of integers or a collection of collections of 2 integers.
        Each element in the inner collection must contain 2 integers, (pooled_rows, pooled_cols)
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, height, width, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:
      The output is the pooled image, flattened across its height and width
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, num_bins, channels)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, channels, num_bins)`.
    """

    @typechecked
    def __init__(
        self,
        bins: Union[Iterable[int], Iterable[Iterable[int]]],
        data_format=None,
        *args,
        **kwargs
    ):
        self.bins = [conv_utils.normalize_tuple(bin, 2, "bin") for bin in bins]
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.pool_layers = []
        for bin in self.bins:
            self.pool_layers.append(AdaptiveAveragePooling2D(bin, self.data_format))
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        dynamic_input_shape = tf.shape(inputs)
        outputs = []
        index = 0
        if self.data_format == "channels_last":
            for bin in self.bins:
                new_inp = inputs[
                    :,
                    : dynamic_input_shape[1] - dynamic_input_shape[1] % bin[0],
                    : dynamic_input_shape[2] - dynamic_input_shape[2] % bin[1],
                    :,
                ]
                output = self.pool_layers[index](new_inp)
                output = tf.reshape(
                    output, [dynamic_input_shape[0], bin[0] * bin[1], inputs.shape[-1]]
                )
                outputs.append(output)
                index += 1
            outputs = tf.concat(outputs, axis=1)
        else:
            for bin in self.bins:
                new_inp = inputs[
                    :,
                    :,
                    : dynamic_input_shape[2] - dynamic_input_shape[2] % bin[0],
                    : dynamic_input_shape[3] - dynamic_input_shape[3] % bin[1],
                ]
                output = self.pool_layers[index](new_inp)
                output = tf.reshape(
                    output, [dynamic_input_shape[0], inputs.shape[1], bin[0] * bin[1]]
                )
                outputs.append(output)
                index += 1

            outputs = tf.concat(outputs, axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        pooled_shape = 0
        for bin in self.bins:
            pooled_shape += tf.reduce_prod(bin)
        if self.data_format == "channels_last":
            return tf.TensorShape([input_shape[0], pooled_shape, input_shape[-1]])
        else:
            return tf.TensorShape([input_shape[0], input_shape[1], pooled_shape])

    def get_config(self):
        config = {"bins": self.bins, "data_format": self.data_format}
        base_config = super().get_config()
        return {**base_config, **config}
