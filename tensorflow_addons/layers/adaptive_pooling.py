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
"""Pooling layers with fixed size outputs"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling2D(tf.keras.layers.Layer):
    """Average Pooling with adaptive kernel size and strides.
        Arguments:
            output_shape: Tuple of integers specifying (Output Height, Output Width)

        Input shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, height, width, features)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, features, height, width)`.

        Output shape:
            - If `data_format='channels_last'`:
                4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
            - If `data_format='channels_first'`:
                4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.

                Here, pooled_rows = output_shape[0], and
                      pooled_cols = output_shape[1]
    """

    def __init__(self, output_shape, data_format="channels_last", **kwargs):
        self.output_image_shape = output_shape
        self.data_format = data_format
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_image_shape[0]
        w_bins = self.output_image_shape[1]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            out_vect = tf.reduce_mean(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = tf.reduce_mean(split_rows, axis=[3, 5])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_image_shape[0],
                    self.output_image_shape[1],
                    input_shape[-1],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_image_shape[0],
                    self.output_image_shape[1],
                ]
            )
        return shape

    def get_config(self):
        config = {
            "output_shape": self.output_image_shape,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
