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
from typeguard import typechecked
from typing import Union, List, Tuple, Callable


class AdaptivePooling1D(tf.keras.layers.Layer):
    """Parent class for 1D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Arguments:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: Integer specifying pooled_features
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: int,
        data_format: str = "channels_last",
        **kwargs
    ):
        if data_format != "channels_first" and data_format != "channels_last":
            raise ValueError(
                "data_format must be one of 'channels_first' or 'channels_last'"
            )
        self.reduce_function = reduce_function
        self.output_size = output_size
        self.data_format = data_format
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        bins = self.output_size
        if self.data_format == "channels_last":
            splits = tf.split(inputs, bins, axis=1)
            splits = tf.stack(splits, axis=1)
            out_vect = self.reduce_function(splits, axis=2)
        else:
            splits = tf.split(inputs, bins, axis=2)
            splits = tf.stack(splits, axis=2)
            out_vect = self.reduce_function(splits, axis=3)
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape([input_shape[0], self.output_size, input_shape[-1]])
        else:
            shape = tf.TensorShape([input_shape[0], input_shape[1], self.output_size])

        return shape


class AdaptivePooling2D(tf.keras.layers.Layer):
    """Parent class for 2D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Arguments:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: Tuple of integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[List[int], Tuple[int, int]],
        data_format: str = "channels_last",
        **kwargs
    ):
        if data_format != "channels_first" and data_format != "channels_last":
            raise ValueError(
                "data_format must be one of 'channels_first' or 'channels_last'"
            )
        self.reduce_function = reduce_function
        self.output_size = output_size
        self.data_format = data_format
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            out_vect = self.reduce_function(split_rows, axis=[2, 4])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            out_vect = self.reduce_function(split_rows, axis=[3, 5])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    input_shape[-1],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                ]
            )

        return shape


class AdaptivePooling3D(tf.keras.layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Arguments:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: Tuple of integers specifying (pooled_height, pooled_width, pooled_depth).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, depth, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width, depth)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[List[int], Tuple[int, int, int]],
        data_format: str = "channels_last",
        **kwargs
    ):
        if data_format != "channels_first" and data_format != "channels_last":
            raise ValueError(
                "data_format must be one of 'channels_first' or 'channels_last'"
            )
        self.reduce_function = reduce_function
        self.output_size = output_size
        self.data_format = data_format
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        h_bins = self.output_size[0]
        w_bins = self.output_size[1]
        d_bins = self.output_size[2]
        if self.data_format == "channels_last":
            split_cols = tf.split(inputs, h_bins, axis=1)
            split_cols = tf.stack(split_cols, axis=1)
            split_rows = tf.split(split_cols, w_bins, axis=3)
            split_rows = tf.stack(split_rows, axis=3)
            split_depth = tf.split(split_rows, d_bins, axis=5)
            split_depth = tf.stack(split_depth, axis=5)
            out_vect = self.reduce_function(split_depth, axis=[2, 4, 6])
        else:
            split_cols = tf.split(inputs, h_bins, axis=2)
            split_cols = tf.stack(split_cols, axis=2)
            split_rows = tf.split(split_cols, w_bins, axis=4)
            split_rows = tf.stack(split_rows, axis=4)
            split_depth = tf.split(split_rows, d_bins, axis=6)
            split_depth = tf.stack(split_depth, axis=6)
            out_vect = self.reduce_function(split_depth, axis=[3, 5, 7])
        return out_vect

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                    input_shape[-1],
                ]
            )
        else:
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                ]
            )

        return shape


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling2D(AdaptivePooling2D):
    """Average Pooling with adaptive kernel size.

    Arguments:
      output_size: Tuple of integers specifying (pooled_rows, pooled_cols).
        The new size of output channels.
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
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.
    """

    @typechecked
    def __init__(
        self,
        output_size: Union[List[int], Tuple[int, int]],
        data_format: str = "channels_last",
        **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}
