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


import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow_addons.layers.deformable_conv2d import DeformableConv2D


def _get_padding_length(
    padding, filter_size, dilation_rate, stride, input_size, output_size
):
    effective_filter_size = (filter_size - 1) * dilation_rate + 1

    pad = 0
    if padding == "same":
        pad = ((output_size - 1) * stride + effective_filter_size - input_size) // 2

    return pad


def _bilinear_interpolate(img, y, x):
    max_height, max_width = img.shape

    if y <= -1 or max_height <= y or x <= -1 or max_width <= x:
        return 0.0

    y_low = int(np.floor(y))
    x_low = int(np.floor(x))
    y_high = y_low + 1
    w_high = x_low + 1

    v1 = 0.0
    if y_low >= 0 and x_low >= 0:
        v1 = img[y_low, x_low]

    v2 = 0.0
    if y_low >= 0 and w_high <= max_width - 1:
        v2 = img[y_low, w_high]

    v3 = 0.0
    if y_high <= max_height - 1 and x_low >= 0:
        v3 = img[y_high, x_low]

    v4 = 0.0
    if y_high <= max_height - 1 and w_high <= max_width - 1:
        v4 = img[y_high, w_high]

    lh = y - y_low
    lw = x - x_low
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


def _expected(
    input_tensor,
    filter_tensor,
    offset_tensor,
    mask_tensor,
    bias,
    strides,
    weight_groups,
    offset_groups,
    padding,
    dilation_rate,
):
    input_tensor = input_tensor.numpy()
    filter_tensor = filter_tensor.numpy()
    offset_tensor = offset_tensor.numpy()
    mask_tensor = mask_tensor.numpy()
    bias = bias.numpy()

    padding = conv_utils.normalize_padding(padding)

    stride_rows, stride_cols = conv_utils.normalize_tuple(strides, 2, "strides")
    dilation_rows, dilation_cols = conv_utils.normalize_tuple(
        dilation_rate, 2, "dilation_rate"
    )
    filter_rows, filter_cols = filter_tensor.shape[-2:]

    batches, input_channels, input_rows, input_cols = input_tensor.shape
    output_channels = filter_tensor.shape[0]

    output_rows = conv_utils.conv_output_length(
        input_rows,
        filter_rows,
        padding=padding,
        stride=stride_rows,
        dilation=dilation_rows,
    )
    output_cols = conv_utils.conv_output_length(
        input_cols,
        filter_cols,
        padding=padding,
        stride=stride_cols,
        dilation=dilation_cols,
    )

    padding_rows = _get_padding_length(
        padding, filter_rows, dilation_rows, stride_rows, input_rows, output_rows
    )
    padding_cols = _get_padding_length(
        padding, filter_cols, dilation_cols, stride_cols, input_cols, output_cols
    )

    input_channels_per_offset_group = input_channels // offset_groups

    input_channels_per_weight_groups = filter_tensor.shape[1]
    output_channels_per_weight_groups = output_channels // weight_groups

    offset_tensor = offset_tensor.reshape((batches, -1, 2, output_rows, output_cols))

    output = np.zeros((batches, output_channels, output_rows, output_cols))

    for batch in range(batches):
        for output_channel in range(output_channels):
            for output_row in range(output_rows):
                for output_col in range(output_cols):
                    for filter_row in range(filter_rows):
                        for filter_col in range(filter_cols):
                            for input_channel in range(
                                input_channels_per_weight_groups
                            ):
                                weight_group = (
                                    output_channel // output_channels_per_weight_groups
                                )
                                current_input_channel = (
                                    weight_group * input_channels_per_weight_groups
                                    + input_channel
                                )

                                offset_group = (
                                    current_input_channel
                                    // input_channels_per_offset_group
                                )
                                offset_idx = (
                                    offset_group * (filter_rows * filter_cols)
                                    + filter_row * filter_cols
                                    + filter_col
                                )

                                dy = offset_tensor[
                                    batch, offset_idx, 0, output_row, output_col
                                ]
                                dx = offset_tensor[
                                    batch, offset_idx, 1, output_row, output_col
                                ]

                                mask = mask_tensor[
                                    batch, offset_idx, output_row, output_col
                                ]

                                y = (
                                    stride_rows * output_row
                                    - padding_rows
                                    + dilation_rows * filter_row
                                    + dy
                                )
                                x = (
                                    stride_cols * output_col
                                    - padding_cols
                                    + dilation_cols * filter_col
                                    + dx
                                )

                                output[
                                    batch, output_channel, output_row, output_col
                                ] += (
                                    mask
                                    * filter_tensor[
                                        output_channel,
                                        input_channel,
                                        filter_row,
                                        filter_col,
                                    ]
                                    * _bilinear_interpolate(
                                        input_tensor[
                                            batch, current_input_channel, :, :
                                        ],
                                        y,
                                        x,
                                    )
                                )

    output += bias.reshape((1, output_channels, 1, 1))
    return output


@pytest.mark.with_device(["cpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_forward_simple(data_format):
    if data_format == "channels_last":
        return

    batches = 1
    input_channels = 6
    filters = 2
    weight_groups = 2
    offset_groups = 3

    strides = (2, 1)
    padding = "same"
    dilation_rate = (2, 1)
    kernel_size = (3, 2)

    input_rows, input_cols = 5, 4
    filter_rows, filter_cols = kernel_size
    stride_rows, stride_cols = strides
    dilation_rows, dilation_cols = dilation_rate

    output_rows = conv_utils.conv_output_length(
        input_rows,
        filter_rows,
        padding=padding,
        stride=stride_rows,
        dilation=dilation_rows,
    )
    output_cols = conv_utils.conv_output_length(
        input_cols,
        filter_cols,
        padding=padding,
        stride=stride_cols,
        dilation=dilation_cols,
    )

    offsets = offset_groups * filter_rows * filter_cols

    input_tensor = tf.random.uniform([batches, input_channels, input_rows, input_cols])
    offset_tensor = tf.random.uniform([batches, 2 * offsets, output_rows, output_cols])
    mask_tensor = tf.random.uniform([batches, offsets, output_rows, output_cols])

    conv = DeformableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        weight_groups=weight_groups,
        offset_groups=offset_groups,
        use_mask=True,
        use_bias=True,
    )

    actual = conv([input_tensor, offset_tensor, mask_tensor])

    filter_tensor = conv.filter_weights
    bias = conv.filter_bias

    expected = _expected(
        input_tensor,
        filter_tensor,
        offset_tensor,
        mask_tensor,
        bias,
        strides,
        weight_groups,
        offset_groups,
        padding,
        dilation_rate,
    )

    np.testing.assert_allclose(actual.numpy(), expected)


@pytest.mark.with_device(["cpu"])
def test_gradients(data_format):
    if data_format == "channels_last":
        return

    batches = 1
    input_channels = 6
    filters = 2
    weight_groups = 2
    offset_groups = 3

    strides = (2, 1)
    padding = "same"
    dilation_rate = (2, 1)
    kernel_size = (3, 2)

    input_rows, input_cols = 5, 4
    filter_rows, filter_cols = kernel_size
    stride_rows, stride_cols = strides
    dilation_rows, dilation_cols = dilation_rate

    output_rows = conv_utils.conv_output_length(
        input_rows,
        filter_rows,
        padding=padding,
        stride=stride_rows,
        dilation=dilation_rows,
    )
    output_cols = conv_utils.conv_output_length(
        input_cols,
        filter_cols,
        padding=padding,
        stride=stride_cols,
        dilation=dilation_cols,
    )

    offsets = offset_groups * filter_rows * filter_cols

    input_tensor = tf.random.uniform([batches, input_channels, input_rows, input_cols])
    offset_tensor = tf.random.uniform([batches, 2 * offsets, output_rows, output_cols])
    mask_tensor = tf.random.uniform([batches, offsets, output_rows, output_cols])

    conv = DeformableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        weight_groups=weight_groups,
        offset_groups=offset_groups,
        use_mask=True,
        use_bias=True,
    )

    def conv_fn(input_tensor, offset_tensor, mask_tensor):
        return conv([input_tensor, offset_tensor, mask_tensor])

    theoretical, numerical = tf.test.compute_gradient(
        conv_fn, [input_tensor, offset_tensor, mask_tensor]
    )

    np.testing.assert_allclose(theoretical[0], numerical[0], atol=1e-3)
    np.testing.assert_allclose(theoretical[1], numerical[1], atol=1e-3)
    np.testing.assert_allclose(theoretical[2], numerical[2], atol=1e-3)


@pytest.mark.with_device(["cpu"])
def test_keras(data_format):
    if data_format == "channels_last":
        return

    batches = 1
    input_channels = 6
    filters = 2
    weight_groups = 2
    offset_groups = 3

    strides = (2, 1)
    padding = "same"
    dilation_rate = (2, 1)
    kernel_size = (3, 2)

    input_rows, input_cols = 5, 4
    filter_rows, filter_cols = kernel_size
    stride_rows, stride_cols = strides
    dilation_rows, dilation_cols = dilation_rate

    output_rows = conv_utils.conv_output_length(
        input_rows,
        filter_rows,
        padding=padding,
        stride=stride_rows,
        dilation=dilation_rows,
    )
    output_cols = conv_utils.conv_output_length(
        input_cols,
        filter_cols,
        padding=padding,
        stride=stride_cols,
        dilation=dilation_cols,
    )

    offsets = offset_groups * filter_rows * filter_cols

    input_tensor = tf.random.uniform([batches, input_channels, input_rows, input_cols])
    offset_tensor = tf.random.uniform([batches, 2 * offsets, output_rows, output_cols])
    mask_tensor = tf.random.uniform([batches, offsets, output_rows, output_cols])

    input_a = tf.keras.Input([input_channels, input_rows, input_cols])
    input_b = tf.keras.Input([2 * offsets, output_rows, output_cols])
    input_c = tf.keras.Input([offsets, output_rows, output_cols])

    conv = DeformableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        weight_groups=weight_groups,
        offset_groups=offset_groups,
        use_mask=True,
        use_bias=True,
    )

    expected_output_shape = tuple(
        conv.compute_output_shape([input_a.shape, input_b.shape, input_c.shape])
    )

    x = [input_a, input_b, input_c]
    y = conv(x)
    model = tf.keras.models.Model(x, y)
    actual_output = model([input_tensor, offset_tensor, mask_tensor])

    assert tf.keras.backend.dtype(y[0]) == "float32"
    assert actual_output.shape[1:] == expected_output_shape[1:]
