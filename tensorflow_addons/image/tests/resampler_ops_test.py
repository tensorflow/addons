# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for resampler."""

import numpy as np
import pytest

import tensorflow as tf
from tensorflow_addons.image import resampler_ops
from tensorflow_addons.utils import test_utils


def _bilinearly_interpolate(data, x, y):
    """Performs bilinenar interpolation of grid data at user defined
    coordinates.

    This interpolation function:
      a) implicitly pads the input data with 0s.
      b) returns 0 when sampling outside the (padded) image.
    The effect is that the sampled signal smoothly goes to 0 outside the
    original input domain, rather than producing a jump discontinuity at
    the image boundaries.
    Args:
      data: numpy array of shape `[data_height, data_width]` containing data
        samples assumed to be defined at the corresponding pixel coordinates.
      x: numpy array of shape `[warp_height, warp_width]` containing
        x coordinates at which interpolation will be performed.
      y: numpy array of shape `[warp_height, warp_width]` containing
        y coordinates at which interpolation will be performed.
    Returns:
      Numpy array of shape `[warp_height, warp_width]` containing interpolated
        values.
    """
    shape = x.shape
    x = np.asarray(x) + 1
    y = np.asarray(y) + 1
    data = np.pad(data, 1, "constant", constant_values=0)

    x_0 = np.floor(x).astype(int)
    x_1 = x_0 + 1
    y_0 = np.floor(y).astype(int)
    y_1 = y_0 + 1

    x_0 = np.clip(x_0, 0, data.shape[1] - 1)
    x_1 = np.clip(x_1, 0, data.shape[1] - 1)
    y_0 = np.clip(y_0, 0, data.shape[0] - 1)
    y_1 = np.clip(y_1, 0, data.shape[0] - 1)

    i_a = data[y_0, x_0]
    i_b = data[y_1, x_0]
    i_c = data[y_0, x_1]
    i_d = data[y_1, x_1]

    w_a = (x_1 - x) * (y_1 - y)
    w_b = (x_1 - x) * (y - y_0)
    w_c = (x - x_0) * (y_1 - y)
    w_d = (x - x_0) * (y - y_0)

    samples = w_a * i_a + w_b * i_b + w_c * i_c + w_d * i_d
    samples = samples.reshape(shape)

    return samples


def _make_warp(batch_size, warp_height, warp_width, dtype):
    """Creates batch of warping coordinates."""
    x, y = np.meshgrid(
        np.linspace(0, warp_width - 1, warp_width),
        np.linspace(0, warp_height - 1, warp_height),
    )
    warp = np.concatenate(
        (
            x.reshape([warp_height, warp_width, 1]),
            y.reshape([warp_height, warp_width, 1]),
        ),
        2,
    )
    warp = np.tile(warp.reshape([1, warp_height, warp_width, 2]), [batch_size, 1, 1, 1])
    warp += np.random.randn(*warp.shape)
    return warp.astype(dtype)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_op_forward_pass(dtype):
    np.random.seed(0)
    data_width = 7
    data_height = 9
    data_channels = 5
    warp_width = 4
    warp_height = 8
    batch_size = 10

    warp = _make_warp(batch_size, warp_height, warp_width, dtype)
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.random.rand(*data_shape).astype(dtype)
    data_ph = tf.constant(data)
    warp_ph = tf.constant(warp)
    outputs = resampler_ops.resampler(data=data_ph, warp=warp_ph)
    assert outputs.shape == (10, warp_height, warp_width, data_channels)

    # Generate reference output via bilinear interpolation in numpy
    reference_output = np.zeros_like(outputs)
    for batch in range(batch_size):
        for c in range(data_channels):
            reference_output[batch, :, :, c] = _bilinearly_interpolate(
                data[batch, :, :, c], warp[batch, :, :, 0], warp[batch, :, :, 1]
            )

    test_utils.assert_allclose_according_to_type(
        outputs, reference_output, half_rtol=5e-3, half_atol=5e-3
    )


def test_op_errors():
    batch_size = 10
    data_height = 9
    data_width = 7
    data_depth = 3
    data_channels = 5
    warp_width = 4
    warp_height = 8

    # Input data shape is not defined over a 2D grid, i.e. its shape is not like
    # (batch_size, data_height, data_width, data_channels).
    data_shape = (batch_size, data_height, data_width, data_depth, data_channels)
    data = np.zeros(data_shape)
    warp_shape = (batch_size, warp_height, warp_width, 2)
    warp = np.zeros(warp_shape)

    with pytest.raises(
        tf.errors.UnimplementedError,
        match="Only bilinear interpolation is currently supported.",
    ):
        resampler_ops.resampler(data, warp)

    # Warp tensor must be at least a matrix, with shape [batch_size, 2].
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.zeros(data_shape)
    warp_shape = (batch_size,)
    warp = np.zeros(warp_shape)

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="warp should be at least a matrix"
    ):
        resampler_ops.resampler(data, warp)

    # The batch size of the data and warp tensors must be the same.
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.zeros(data_shape)
    warp_shape = (batch_size + 1, warp_height, warp_width, 2)
    warp = np.zeros(warp_shape)

    with pytest.raises(
        tf.errors.InvalidArgumentError, match="Batch size of data and warp tensor"
    ):
        resampler_ops.resampler(data, warp)

    # The warp tensor must contain 2D coordinates, i.e. its shape last dimension
    # must be 2.
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.zeros(data_shape)
    warp_shape = (batch_size, warp_height, warp_width, 3)
    warp = np.zeros(warp_shape)

    with pytest.raises(
        tf.errors.UnimplementedError,
        match="Only bilinear interpolation is supported, warping",
    ):
        resampler_ops.resampler(data, warp)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_op_backward_pass(dtype):
    np.random.seed(13)
    data_width = 5
    data_height = 4
    data_channels = 3
    warp_width = 2
    warp_height = 6
    batch_size = 3

    warp = _make_warp(batch_size, warp_height, warp_width, dtype)
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.random.rand(*data_shape).astype(dtype)
    data_tensor = tf.constant(data)
    warp_tensor = tf.constant(warp)
    theoretical, _ = tf.test.compute_gradient(
        resampler_ops.resampler, [data_tensor, warp_tensor]
    )
    data_tensor_64 = tf.constant(data, dtype=tf.float64)
    warp_tensor_64 = tf.constant(warp, dtype=tf.float64)
    _, numerical_64 = tf.test.compute_gradient(
        resampler_ops.resampler, [data_tensor_64, warp_tensor_64]
    )

    for t, n in zip(theoretical, numerical_64):
        test_utils.assert_allclose_according_to_type(
            t, n, float_rtol=5e-5, float_atol=5e-5
        )


@pytest.mark.with_device(["cpu", "gpu"])
def test_op_empty_batch():
    np.random.seed(13)
    data_width = 5
    data_height = 4
    data_channels = 3
    warp_width = 2
    warp_height = 6
    batch_size = 0
    dtype = np.float32

    warp = _make_warp(batch_size, warp_height, warp_width, dtype)
    data_shape = (batch_size, data_height, data_width, data_channels)
    data = np.zeros(data_shape).astype(dtype)
    data_tensor = tf.constant(data)
    warp_tensor = tf.constant(warp)
    with tf.GradientTape() as tape:
        tape.watch(data_tensor)
        tape.watch(warp_tensor)
        outputs = resampler_ops.resampler(data=data_tensor, warp=warp_tensor)
    data_grad, warp_grad = tape.gradient(outputs, (data_tensor, warp_tensor))
    assert data_grad.shape == data.shape
    assert warp_grad.shape == warp.shape
