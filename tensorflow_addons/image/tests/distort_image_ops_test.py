# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may noa use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for python distort_image_ops."""

import pytest
import numpy as np

import tensorflow as tf
from tensorflow_addons.image import distort_image_ops
from tensorflow_addons.utils import test_utils


def _adjust_hue_in_yiq_np(x_np, delta_h):
    """Rotate hue in YIQ space.

    Mathematically we first convert rgb color to yiq space, rotate the hue
    degrees, and then convert back to rgb.

    Args:
        x_np: input x with last dimension = 3.
        delta_h: degree of hue rotation, in radians.

    Returns:
        Adjusted y with the same shape as x_np.
    """
    assert x_np.shape[-1] == 3
    x_v = x_np.reshape([-1, 3])
    u = np.cos(delta_h)
    w = np.sin(delta_h)
    # Projection matrix from RGB to YIQ. Numbers from wikipedia
    # https://en.wikipedia.org/wiki/YIQ
    tyiq = np.array(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]
    ).astype(x_v.dtype)
    inverse_tyiq = np.array(
        [
            [1.0, 0.95617069, 0.62143257],
            [1.0, -0.2726886, -0.64681324],
            [1.0, -1.103744, 1.70062309],
        ]
    ).astype(x_v.dtype)
    y_v = np.dot(x_v, tyiq.T).astype(x_v.dtype)
    # Hue rotation matrix in YIQ space.
    hue_rotation = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]]).astype(
        x_v.dtype
    )
    y_v = np.dot(y_v, hue_rotation.T)
    # Projecting back to RGB space.
    y_v = np.dot(y_v, inverse_tyiq.T)
    return y_v.reshape(x_np.shape)


def _adjust_hue_in_yiq_tf(x_np, delta_h):
    x = tf.constant(x_np)
    y = distort_image_ops.adjust_hsv_in_yiq(x, delta_h, 1, 1)
    return y


@pytest.mark.parametrize(
    "shape", ([2, 2, 3], [4, 2, 3], [2, 4, 3], [2, 5, 3], [1000, 1, 3])
)
@pytest.mark.parametrize(
    "style", ("all_random", "rg_same", "rb_same", "gb_same", "rgb_same")
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_adjust_random_hue_in_yiq(shape, style, dtype):
    x_np = (np.random.rand(*shape) * 255.0).astype(dtype)
    delta_h = (np.random.rand() * 2.0 - 1.0) * np.pi
    if style == "all_random":
        pass
    elif style == "rg_same":
        x_np[..., 1] = x_np[..., 0]
    elif style == "rb_same":
        x_np[..., 2] = x_np[..., 0]
    elif style == "gb_same":
        x_np[..., 2] = x_np[..., 1]
    elif style == "rgb_same":
        x_np[..., 1] = x_np[..., 0]
        x_np[..., 2] = x_np[..., 0]
    else:
        raise AssertionError("Invalid test style: %s" % (style))
    y_np = _adjust_hue_in_yiq_np(x_np, delta_h)
    y_tf = _adjust_hue_in_yiq_tf(x_np, delta_h)
    test_utils.assert_allclose_according_to_type(
        y_tf, y_np, atol=1e-4, rtol=2e-4, half_rtol=0.8
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_invalid_rank_hsv():
    x_np = np.random.rand(2, 3) * 255.0
    delta_h = np.random.rand() * 2.0 - 1.0
    with pytest.raises(
        (tf.errors.InvalidArgumentError, ValueError), match="input must be at least 3-D"
    ):
        _adjust_hue_in_yiq_tf(x_np, delta_h)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_invalid_channels_hsv():
    x_np = np.random.rand(4, 2, 4) * 255.0
    delta_h = np.random.rand() * 2.0 - 1.0
    with pytest.raises(
        (tf.errors.InvalidArgumentError, ValueError),
        match="input must have 3 channels but instead has 4",
    ):
        _adjust_hue_in_yiq_tf(x_np, delta_h)


def test_adjust_hsv_in_yiq_unknown_shape():
    fn = tf.function(distort_image_ops.adjust_hsv_in_yiq).get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float64)
    )
    for shape in (2, 3, 3), (4, 2, 3, 3):
        image_np = np.random.rand(*shape) * 255.0
        image_tf = tf.constant(image_np)
        np.testing.assert_allclose(
            _adjust_hue_in_yiq_np(image_np, 0), fn(image_tf), rtol=2e-4, atol=1e-4
        )


def test_random_hsv_in_yiq_unknown_shape():
    fn = tf.function(distort_image_ops.random_hsv_in_yiq).get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float32)
    )
    for shape in (2, 3, 3), (4, 2, 3, 3):
        image_tf = tf.ones(shape)
        np.testing.assert_equal(fn(image_tf).numpy(), fn(image_tf).numpy())


def _adjust_value_in_yiq_np(x_np, scale):
    return x_np * scale


def _adjust_value_in_yiq_tf(x_np, scale):
    x = tf.constant(x_np)
    y = distort_image_ops.adjust_hsv_in_yiq(x, 0, 1, scale)
    return y


def test_adjust_random_value_in_yiq():
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    for x_shape in x_shapes:
        for test_style in test_styles:
            x_np = np.random.rand(*x_shape) * 255.0
            scale = np.random.rand() * 2.0 - 1.0
            if test_style == "all_random":
                pass
            elif test_style == "rg_same":
                x_np[..., 1] = x_np[..., 0]
            elif test_style == "rb_same":
                x_np[..., 2] = x_np[..., 0]
            elif test_style == "gb_same":
                x_np[..., 2] = x_np[..., 1]
            elif test_style == "rgb_same":
                x_np[..., 1] = x_np[..., 0]
                x_np[..., 2] = x_np[..., 0]
            else:
                raise AssertionError("Invalid test style: %s" % (test_style))
            y_np = _adjust_value_in_yiq_np(x_np, scale)
            y_tf = _adjust_value_in_yiq_tf(x_np, scale)
            np.testing.assert_allclose(y_tf, y_np, rtol=2e-4, atol=1e-4)


def test_invalid_rank_value():
    x_np = np.random.rand(2, 3) * 255.0
    scale = np.random.rand() * 2.0 - 1.0
    if tf.executing_eagerly():
        with pytest.raises(
            (tf.errors.InvalidArgumentError, ValueError),
            match="input must be at least 3-D",
        ):
            _adjust_value_in_yiq_tf(x_np, scale)
    else:
        with pytest.raises(
            ValueError, match="Shape must be at least rank 3 but is rank 2"
        ):
            _adjust_value_in_yiq_tf(x_np, scale)


def test_invalid_channels_value():
    x_np = np.random.rand(4, 2, 4) * 255.0
    scale = np.random.rand() * 2.0 - 1.0
    if tf.executing_eagerly():
        with pytest.raises(
            (tf.errors.InvalidArgumentError, ValueError),
            match="input must have 3 channels but instead has 4",
        ):
            _adjust_value_in_yiq_tf(x_np, scale)
    else:
        with pytest.raises(ValueError, match="Dimension must be 3 but is 4"):
            _adjust_value_in_yiq_tf(x_np, scale)


def _adjust_saturation_in_yiq_tf(x_np, scale):
    x = tf.constant(x_np)
    y = distort_image_ops.adjust_hsv_in_yiq(x, 0, scale, 1)
    return y


def _adjust_saturation_in_yiq_np(x_np, scale):
    """Adjust saturation using linear interpolation."""
    rgb_weights = np.array([0.299, 0.587, 0.114])
    gray = np.sum(x_np * rgb_weights, axis=-1, keepdims=True)
    y_v = x_np * scale + gray * (1 - scale)
    return y_v


@pytest.mark.parametrize(
    "shape", ([2, 2, 3], [4, 2, 3], [2, 4, 3], [2, 5, 3], [1000, 1, 3])
)
@pytest.mark.parametrize(
    "style", ("all_random", "rg_same", "rb_same", "gb_same", "rgb_same")
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_adjust_random_saturation_in_yiq(shape, style, dtype):
    x_np = (np.random.rand(*shape) * 255.0).astype(dtype)
    scale = np.random.rand() * 2.0 - 1.0
    if style == "all_random":
        pass
    elif style == "rg_same":
        x_np[..., 1] = x_np[..., 0]
    elif style == "rb_same":
        x_np[..., 2] = x_np[..., 0]
    elif style == "gb_same":
        x_np[..., 2] = x_np[..., 1]
    elif style == "rgb_same":
        x_np[..., 1] = x_np[..., 0]
        x_np[..., 2] = x_np[..., 0]
    else:
        raise AssertionError("Invalid test style: %s" % (style))
    y_baseline = _adjust_saturation_in_yiq_np(x_np, scale)
    y_tf = _adjust_saturation_in_yiq_tf(x_np, scale)
    test_utils.assert_allclose_according_to_type(
        y_tf, y_baseline, atol=1e-4, rtol=2e-4, half_rtol=0.8
    )


def test_invalid_rank():
    x_np = np.random.rand(2, 3) * 255.0
    scale = np.random.rand() * 2.0 - 1.0

    msg = "input must be at least 3-D"
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError), match=msg):
        _adjust_saturation_in_yiq_tf(x_np, scale).numpy()


def test_invalid_channels():
    x_np = np.random.rand(4, 2, 4) * 255.0
    scale = np.random.rand() * 2.0 - 1.0
    msg = "input must have 3 channels but instead has 4"
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError), match=msg):
        _adjust_saturation_in_yiq_tf(x_np, scale).numpy()
