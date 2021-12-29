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


import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.optical_flow import CorrelationCost


def _forward(
    input_a,
    input_b,
    kernel_size,
    max_displacement,
    stride_1,
    stride_2,
    pad,
    data_format,
):
    input_a_op = tf.convert_to_tensor(input_a, dtype=tf.float32)
    input_b_op = tf.convert_to_tensor(input_b, dtype=tf.float32)

    output = CorrelationCost(
        kernel_size=kernel_size,
        max_displacement=max_displacement,
        stride_1=stride_1,
        stride_2=stride_2,
        pad=pad,
        data_format=data_format,
    )([input_a_op, input_b_op])

    return output


def _create_test_data(data_format):
    # Produce test data for _forward_simple and _keras methods
    val_a = np.array(
        [
            [
                [[0, -6, 9, 5], [1, -5, 10, 3], [2, -4, 11, 1]],
                [[3, -3, 12, -1], [4, -2, 13, -3], [5, -1, 14, -5]],
            ],
            [
                [[6, 0, 15, -7], [7, 1, 16, -9], [8, 2, 17, -11]],
                [[9, 3, 18, -13], [10, 4, 19, -15], [11, 5, 20, -17]],
            ],
        ],
        dtype=np.float32,
    )

    # pylint: disable=too-many-function-args
    val_b = val_a.transpose(2, 3, 0, 1).reshape(2, 2, 3, 4)
    # pylint: enable=too-many-function-args

    if data_format == "channels_last":
        val_a = np.moveaxis(val_a, 1, -1)
        val_b = np.moveaxis(val_b, 1, -1)

    return val_a, val_b


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_forward_simple(data_format):
    # We are just testing where the output has vanishing values.
    val_a, val_b = _create_test_data(data_format)
    input_a = tf.constant(val_a, dtype=tf.float32)
    input_b = tf.constant(val_b, dtype=tf.float32)

    input_a_tensor = tf.convert_to_tensor(input_a, dtype=tf.float32)
    input_b_tensor = tf.convert_to_tensor(input_b, dtype=tf.float32)

    kernel_size = 1
    max_displacement = 2
    stride_1 = 1
    stride_2 = 2
    pad = 4

    actual = _forward(
        input_a_tensor,
        input_b_tensor,
        kernel_size=kernel_size,
        max_displacement=max_displacement,
        stride_1=stride_1,
        stride_2=stride_2,
        pad=pad,
        data_format=data_format,
    )

    if data_format == "channels_last":
        # NHWC -> NCHW
        actual = tf.transpose(actual, [0, 3, 1, 2])

    # We can test fixed ids, as output is independent from data_format
    expected_ids = np.concatenate(
        [
            np.zeros(464),
            np.ones(464),
        ]
    )
    np.testing.assert_allclose(tf.where(actual == 0)[:, 0].numpy(), expected_ids)

    counts = [54, 52, 54, 50, 44, 50, 54, 52, 54]
    expected_ids = np.concatenate([k * np.ones(v) for k, v in enumerate(counts)])
    expected_ids = np.concatenate([expected_ids, expected_ids])
    np.testing.assert_allclose(tf.where(actual == 0)[:, 1], expected_ids)
    assert actual.shape == (2, 9, 7, 8)


@pytest.mark.with_device(["cpu", "gpu"])
def test_gradients(data_format):
    batch, channels, height, width = 2, 3, 5, 6
    input_a = np.random.randn(batch, channels, height, width).astype(np.float32)
    input_b = np.random.randn(batch, channels, height, width).astype(np.float32)

    kernel_size = 1
    max_displacement = 2
    stride_1 = 1
    stride_2 = 2
    pad = 4

    if data_format == "channels_last":
        input_a = tf.transpose(input_a, [0, 2, 3, 1])
        input_b = tf.transpose(input_b, [0, 2, 3, 1])

    input_a_op = tf.convert_to_tensor(input_a)
    input_b_op = tf.convert_to_tensor(input_b)

    def correlation_fn(input_a, input_b):
        return CorrelationCost(
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride_1=stride_1,
            stride_2=stride_2,
            pad=pad,
            data_format=data_format,
        )([input_a, input_b])

    theoretical, numerical = tf.test.compute_gradient(
        correlation_fn, [input_a_op, input_b_op]
    )

    np.testing.assert_allclose(theoretical[0], numerical[0], atol=1e-3)


@pytest.mark.with_device(["cpu", "gpu"])
def test_keras(data_format):
    # Unable to use `layer_test` as this layer has multiple inputs.
    val_a, val_b = _create_test_data(data_format)

    input_a = tf.keras.Input(shape=val_a.shape[1:])
    input_b = tf.keras.Input(shape=val_b.shape[1:])

    layer = CorrelationCost(
        kernel_size=1,
        max_displacement=2,
        stride_1=1,
        stride_2=2,
        pad=4,
        data_format=data_format,
    )

    expected_output_shape = tuple(
        layer.compute_output_shape([input_a.shape, input_b.shape])
    )

    x = [input_a, input_b]
    y = layer(x)
    model = tf.keras.models.Model(x, y)
    actual_output = model([val_a, val_b])

    expected_output_type = "float32"
    assert tf.keras.backend.dtype(y[0]) == expected_output_type
    assert actual_output.shape[1:] == expected_output_shape[0][1:]
