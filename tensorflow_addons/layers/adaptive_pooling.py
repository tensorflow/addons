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
import tensorflow_addons.utils.keras_utils as conv_utils

from typeguard import typechecked
from typing import Union, Callable, Iterable


class AdaptivePooling1D(tf.keras.layers.Layer):
    """Parent class for 1D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of a single integer, specifying pooled_features.
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
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = conv_utils.normalize_tuple(output_size, 1, "output_size")
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        if self.data_format == "channels_last":
            start_points = tf.cast(
                (
                    tf.range(self.output_size[0], dtype=tf.float32)
                    * (inputs.shape[1] / self.output_size[0])
                ),
                tf.int32,
            )
            end_points = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size[0], dtype=tf.float32) + 1)
                        * (inputs.shape[1] / self.output_size[0])
                    )
                ),
                tf.int32,
            )
            pooled = []
            for idx in range(self.output_size[0]):
                pooled.append(
                    self.reduce_function(
                        inputs[:, start_points[idx] : end_points[idx], :],
                        axis=1,
                        keepdims=True,
                    )
                )
            pooled = tf.concat(pooled, axis=1)
        else:
            start_points = tf.cast(
                (
                    tf.range(self.output_size, dtype=tf.float32)
                    * (inputs.shape[2] / self.output_size[0])
                ),
                tf.int32,
            )
            end_points = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size, dtype=tf.float32) + 1)
                        * (inputs.shape[2] / self.output_size[0])
                    )
                ),
                tf.int32,
            )
            pooled = []
            for idx in range(self.output_size[0]):
                pooled.append(
                    self.reduce_function(
                        inputs[:, :, start_points[idx] : end_points[idx]],
                        axis=2,
                        keepdims=True,
                    )
                )
            pooled = tf.concat(pooled, axis=2)

        return pooled

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [input_shape[0], self.output_size[0], input_shape[2]]
            )
        else:
            shape = tf.TensorShape(
                [input_shape[0], input_shape[1], self.output_size[0]]
            )

        return shape

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling1D(AdaptivePooling1D):
    """Average Pooling with adaptive kernel size.

    Args:
      output_size: An integer or tuple/list of a single integer, specifying pooled_features.
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, steps)`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch, steps, channels)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch, channels, steps)`.

    Output shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, pooled_steps, channels)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, channels, pooled_steps)`.
    """

    @typechecked
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveMaxPooling1D(AdaptivePooling1D):
    """Max Pooling with adaptive kernel size.

    Args:
      output_size: An integer or tuple/list of a single integer, specifying pooled_features.
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, steps)`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch, steps, channels)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch, channels, steps)`.

    Output shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, pooled_steps, channels)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, channels, pooled_steps)`.
    """

    @typechecked
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_max, output_size, data_format, **kwargs)


class AdaptivePooling2D(tf.keras.layers.Layer):
    """Parent class for 2D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 2 integers specifying (pooled_rows, pooled_cols).
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
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = conv_utils.normalize_tuple(output_size, 2, "output_size")
        self.output_size_x, self.output_size_y = self.output_size
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        if self.data_format == "channels_last":
            start_points_x = tf.cast(
                (
                    tf.range(self.output_size_x, dtype=tf.float32)
                    * (inputs.shape[1] / self.output_size_x)
                ),
                tf.int32,
            )
            end_points_x = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_x, dtype=tf.float32) + 1)
                        * (inputs.shape[1] / self.output_size_x)
                    )
                ),
                tf.int32,
            )

            start_points_y = tf.cast(
                (
                    tf.range(self.output_size_y, dtype=tf.float32)
                    * (inputs.shape[2] / self.output_size_y)
                ),
                tf.int32,
            )
            end_points_y = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_y, dtype=tf.float32) + 1)
                        * (inputs.shape[2] / self.output_size_y)
                    )
                ),
                tf.int32,
            )
            pooled = []
            for idx in range(self.output_size_x):
                pooled.append(
                    self.reduce_function(
                        inputs[:, start_points_x[idx] : end_points_x[idx], :, :],
                        axis=1,
                        keepdims=True,
                    )
                )
            x_pooled = tf.concat(pooled, axis=1)

            pooled = []
            for idx in range(self.output_size_y):
                pooled.append(
                    self.reduce_function(
                        x_pooled[:, :, start_points_y[idx] : end_points_y[idx], :],
                        axis=2,
                        keepdims=True,
                    )
                )
            y_pooled = tf.concat(pooled, axis=2)

        else:
            start_points_x = tf.cast(
                (
                    tf.range(self.output_size_x, dtype=tf.float32)
                    * (inputs.shape[2] / self.output_size_x)
                ),
                tf.int32,
            )
            end_points_x = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_x, dtype=tf.float32) + 1)
                        * (inputs.shape[2] / self.output_size_x)
                    )
                ),
                tf.int32,
            )

            start_points_y = tf.cast(
                (
                    tf.range(self.output_size_y, dtype=tf.float32)
                    * (inputs.shape[3] / self.output_size_y)
                ),
                tf.int32,
            )
            end_points_y = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_y, dtype=tf.float32) + 1)
                        * (inputs.shape[3] / self.output_size_y)
                    )
                ),
                tf.int32,
            )
            pooled = []
            for idx in range(self.output_size_x):
                pooled.append(
                    self.reduce_function(
                        inputs[:, :, start_points_x[idx] : end_points_x[idx], :],
                        axis=2,
                        keepdims=True,
                    )
                )
            x_pooled = tf.concat(pooled, axis=2)

            pooled = []
            for idx in range(self.output_size_y):
                pooled.append(
                    self.reduce_function(
                        x_pooled[:, :, :, start_points_y[idx] : end_points_y[idx]],
                        axis=3,
                        keepdims=True,
                    )
                )
            y_pooled = tf.concat(pooled, axis=3)

        return y_pooled

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    input_shape[3],
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

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling2D(AdaptivePooling2D):
    """Average Pooling with adaptive kernel size.

    Args:
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
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveMaxPooling2D(AdaptivePooling2D):
    """Max Pooling with adaptive kernel size.

    Args:
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
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_max, output_size, data_format, **kwargs)


class AdaptivePooling3D(tf.keras.layers.Layer):
    """Parent class for 3D pooling layers with adaptive kernel size.

    This class only exists for code reuse. It will never be an exposed API.

    Args:
      reduce_function: The reduction method to apply, e.g. `tf.reduce_max`.
      output_size: An integer or tuple/list of 3 integers specifying (pooled_dim1, pooled_dim2, pooled_dim3).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    """

    @typechecked
    def __init__(
        self,
        reduce_function: Callable,
        output_size: Union[int, Iterable[int]],
        data_format=None,
        **kwargs,
    ):
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.reduce_function = reduce_function
        self.output_size = conv_utils.normalize_tuple(output_size, 3, "output_size")
        self.output_size_x, self.output_size_y, self.output_size_z = self.output_size
        super().__init__(**kwargs)

    def call(self, inputs, *args):
        if self.data_format == "channels_last":
            start_points_x = tf.cast(
                (
                    tf.range(self.output_size_x, dtype=tf.float32)
                    * (inputs.shape[1] / self.output_size_x)
                ),
                tf.int32,
            )
            end_points_x = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_x, dtype=tf.float32) + 1)
                        * (inputs.shape[1] / self.output_size_x)
                    )
                ),
                tf.int32,
            )

            start_points_y = tf.cast(
                (
                    tf.range(self.output_size_y, dtype=tf.float32)
                    * (inputs.shape[2] / self.output_size_y)
                ),
                tf.int32,
            )
            end_points_y = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_y, dtype=tf.float32) + 1)
                        * (inputs.shape[2] / self.output_size_y)
                    )
                ),
                tf.int32,
            )

            start_points_z = tf.cast(
                (
                    tf.range(self.output_size_z, dtype=tf.float32)
                    * (inputs.shape[3] / self.output_size_z)
                ),
                tf.int32,
            )
            end_points_z = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_z, dtype=tf.float32) + 1)
                        * (inputs.shape[3] / self.output_size_z)
                    )
                ),
                tf.int32,
            )

            pooled = []
            for idx in range(self.output_size_x):
                pooled.append(
                    self.reduce_function(
                        inputs[:, start_points_x[idx] : end_points_x[idx], :, :, :],
                        axis=1,
                        keepdims=True,
                    )
                )
            x_pooled = tf.concat(pooled, axis=1)

            pooled = []
            for idx in range(self.output_size_y):
                pooled.append(
                    self.reduce_function(
                        x_pooled[:, :, start_points_y[idx] : end_points_y[idx], :, :],
                        axis=2,
                        keepdims=True,
                    )
                )
            y_pooled = tf.concat(pooled, axis=2)

            pooled = []
            for idx in range(self.output_size_z):
                pooled.append(
                    self.reduce_function(
                        y_pooled[:, :, :, start_points_z[idx] : end_points_z[idx], :],
                        axis=3,
                        keepdims=True,
                    )
                )
            z_pooled = tf.concat(pooled, axis=3)

        else:
            start_points_x = tf.cast(
                (
                    tf.range(self.output_size_x, dtype=tf.float32)
                    * (inputs.shape[2] / self.output_size_x)
                ),
                tf.int32,
            )
            end_points_x = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_x, dtype=tf.float32) + 1)
                        * (inputs.shape[2] / self.output_size_x)
                    )
                ),
                tf.int32,
            )

            start_points_y = tf.cast(
                (
                    tf.range(self.output_size_y, dtype=tf.float32)
                    * (inputs.shape[3] / self.output_size_y)
                ),
                tf.int32,
            )
            end_points_y = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_y, dtype=tf.float32) + 1)
                        * (inputs.shape[3] / self.output_size_y)
                    )
                ),
                tf.int32,
            )

            start_points_z = tf.cast(
                (
                    tf.range(self.output_size_z, dtype=tf.float32)
                    * (inputs.shape[4] / self.output_size_z)
                ),
                tf.int32,
            )
            end_points_z = tf.cast(
                tf.math.ceil(
                    (
                        (tf.range(self.output_size_z, dtype=tf.float32) + 1)
                        * (inputs.shape[4] / self.output_size_z)
                    )
                ),
                tf.int32,
            )

            pooled = []
            for idx in range(self.output_size_x):
                pooled.append(
                    self.reduce_function(
                        inputs[:, :, start_points_x[idx] : end_points_x[idx], :, :],
                        axis=2,
                        keepdims=True,
                    )
                )
            x_pooled = tf.concat(pooled, axis=2)

            pooled = []
            for idx in range(self.output_size_y):
                pooled.append(
                    self.reduce_function(
                        x_pooled[:, :, :, start_points_y[idx] : end_points_y[idx], :],
                        axis=3,
                        keepdims=True,
                    )
                )
            y_pooled = tf.concat(pooled, axis=3)

            pooled = []
            for idx in range(self.output_size_z):
                pooled.append(
                    self.reduce_function(
                        y_pooled[:, :, :, :, start_points_z[idx] : end_points_z[idx]],
                        axis=4,
                        keepdims=True,
                    )
                )
            z_pooled = tf.concat(pooled, axis=4)

        return z_pooled

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            shape = tf.TensorShape(
                [
                    input_shape[0],
                    self.output_size[0],
                    self.output_size[1],
                    self.output_size[2],
                    input_shape[4],
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

    def get_config(self):
        config = {
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveAveragePooling3D(AdaptivePooling3D):
    """Average Pooling with adaptive kernel size.

    Args:
      output_size: An integer or tuple/list of 3 integers specifying (pooled_depth, pooled_height, pooled_width).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.
    """

    @typechecked
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_mean, output_size, data_format, **kwargs)


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaptiveMaxPooling3D(AdaptivePooling3D):
    """Max Pooling with adaptive kernel size.

    Args:
      output_size: An integer or tuple/list of 3 integers specifying (pooled_depth, pooled_height, pooled_width).
        The new size of output channels.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.
    """

    @typechecked
    def __init__(
        self, output_size: Union[int, Iterable[int]], data_format=None, **kwargs
    ):
        super().__init__(tf.reduce_max, output_size, data_format, **kwargs)
