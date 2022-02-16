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
"""Tests for transform ops."""

import pytest
import numpy as np
import tensorflow as tf

from skimage import transform

from tensorflow_addons.image import transform_ops
from tensorflow_addons.utils import test_utils

_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_compose(dtype):
    image = tf.constant(
        [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=dtype
    )
    # Rotate counter-clockwise by pi / 2.
    rotation = transform_ops.angles_to_projective_transforms(np.pi / 2, 4, 4)
    # Translate right by 1 (the transformation matrix is always inverted,
    # hence the -1).
    translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0], dtype=tf.dtypes.float32)
    composed = transform_ops.compose_transforms([rotation, translation])
    image_transformed = transform_ops.transform(image, composed)
    np.testing.assert_equal(
        [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1]],
        image_transformed.numpy(),
    )


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_extreme_projective_transform(dtype):
    image = tf.constant(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtype
    )
    transformation = tf.constant([1, 0, 0, 0, 1, 0, -1, 0], tf.dtypes.float32)
    image_transformed = transform_ops.transform(image, transformation)
    np.testing.assert_equal(
        [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        image_transformed.numpy(),
    )


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("fill_value", [0.0, 1.0])
def test_transform_constant_fill_mode(dtype, fill_value):
    image = tf.constant(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=dtype
    )
    expected = np.asarray(
        [
            [fill_value, 0, 1, 2],
            [fill_value, 4, 5, 6],
            [fill_value, 8, 9, 10],
            [fill_value, 12, 13, 14],
        ],
        dtype=dtype.as_numpy_dtype,
    )
    # Translate right by 1 (the transformation matrix is always inverted,
    # hence the -1).
    translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0], dtype=tf.float32)
    image_transformed = transform_ops.transform(
        image,
        translation,
        fill_mode="constant",
        fill_value=fill_value,
    )
    np.testing.assert_equal(image_transformed.numpy(), expected)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_transform_reflect_fill_mode(dtype):
    image = tf.constant(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=dtype
    )
    expected = np.asarray(
        [[0, 0, 1, 2], [4, 4, 5, 6], [8, 8, 9, 10], [12, 12, 13, 14]],
        dtype=dtype.as_numpy_dtype,
    )
    # Translate right by 1 (the transformation matrix is always inverted,
    # hence the -1).
    translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0], dtype=tf.float32)
    image_transformed = transform_ops.transform(image, translation, fill_mode="reflect")
    np.testing.assert_equal(image_transformed.numpy(), expected)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_transform_wrap_fill_mode(dtype):
    image = tf.constant(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=dtype
    )
    expected = np.asarray(
        [[3, 0, 1, 2], [7, 4, 5, 6], [11, 8, 9, 10], [15, 12, 13, 14]],
        dtype=dtype.as_numpy_dtype,
    )
    # Translate right by 1 (the transformation matrix is always inverted,
    # hence the -1).
    translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0], dtype=tf.float32)
    image_transformed = transform_ops.transform(image, translation, fill_mode="wrap")
    np.testing.assert_equal(image_transformed.numpy(), expected)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_transform_nearest_fill_mode(dtype):
    image = tf.constant(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=dtype
    )
    expected = np.asarray(
        [[0, 0, 0, 1], [4, 4, 4, 5], [8, 8, 8, 9], [12, 12, 12, 13]],
        dtype=dtype.as_numpy_dtype,
    )
    # Translate right by 2 (the transformation matrix is always inverted,
    # hence the -2).
    translation = tf.constant([1, 0, -2, 0, 1, 0, 0, 0], dtype=tf.float32)
    image_transformed = transform_ops.transform(image, translation, fill_mode="nearest")
    np.testing.assert_equal(image_transformed.numpy(), expected)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_transform_static_output_shape():
    image = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    result = transform_ops.transform(
        image, tf.random.uniform([8], -1, 1), output_shape=[3, 5]
    )
    np.testing.assert_equal([3, 5], result.shape)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_transform_unknown_shape():
    fn = tf.function(transform_ops.transform).get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float32), [1, 0, 0, 0, 1, 0, 0, 0]
    )
    for shape in (2, 4), (2, 4, 3), (1, 2, 4, 3):
        image = tf.ones(shape=shape)
        np.testing.assert_equal(image.numpy(), fn(image).numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def _test_grad(input_shape, output_shape=None):
    image_size = tf.math.cumprod(input_shape)[-1]
    image_size = tf.cast(image_size, tf.float32)
    test_image = tf.reshape(tf.range(0, image_size, dtype=tf.float32), input_shape)
    # Scale test image to range [0, 0.01]
    test_image = (test_image / image_size) * 0.01

    def transform_fn(x):
        x.set_shape(input_shape)
        transform = transform_ops.angles_to_projective_transforms(np.pi / 2, 4, 4)
        return transform_ops.transform(
            images=x, transforms=transform, output_shape=output_shape
        )

    theoretical, numerical = tf.test.compute_gradient(transform_fn, [test_image])

    np.testing.assert_almost_equal(theoretical[0], numerical[0], decimal=6)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_grad():
    _test_grad([8, 8])
    _test_grad([8, 8], [4, 4])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_transform_data_types(dtype):
    image = tf.constant([[1, 2], [3, 4]], dtype=dtype)
    np.testing.assert_equal(
        np.array([[4, 4], [4, 4]]).astype(dtype.as_numpy_dtype),
        transform_ops.transform(image, [1] * 8),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_transform_eager():
    image = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_equal(
        np.array([[4, 4], [4, 4]]), transform_ops.transform(image, [1] * 8)
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_zeros(dtype):
    for shape in [(5, 5), (24, 24), (2, 24, 24, 3)]:
        for angle in [0, 1, np.pi / 2.0]:
            image = tf.zeros(shape, dtype)
            np.testing.assert_equal(
                transform_ops.rotate(image, angle),
                np.zeros(shape, dtype.as_numpy_dtype),
            )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_rotate_even(dtype):
    image = tf.reshape(tf.cast(tf.range(36), dtype), (6, 6))
    image_rep = tf.tile(image[None, :, :, None], [3, 1, 1, 1])
    angles = tf.constant([0.0, np.pi / 4.0, np.pi / 2.0], tf.float32)
    image_rotated = transform_ops.rotate(image_rep, angles)
    np.testing.assert_equal(
        image_rotated.numpy()[:, :, :, 0],
        [
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35],
            ],
            [
                [0, 3, 4, 11, 17, 0],
                [2, 3, 9, 16, 23, 23],
                [1, 8, 15, 21, 22, 29],
                [6, 13, 20, 21, 27, 34],
                [12, 18, 19, 26, 33, 33],
                [0, 18, 24, 31, 32, 0],
            ],
            [
                [5, 11, 17, 23, 29, 35],
                [4, 10, 16, 22, 28, 34],
                [3, 9, 15, 21, 27, 33],
                [2, 8, 14, 20, 26, 32],
                [1, 7, 13, 19, 25, 31],
                [0, 6, 12, 18, 24, 30],
            ],
        ],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_rotate_odd(dtype):
    image = tf.reshape(tf.cast(tf.range(25), dtype), (5, 5))
    image_rep = tf.tile(image[None, :, :, None], [3, 1, 1, 1])
    angles = tf.constant([np.pi / 4.0, 1.0, -np.pi / 2.0], tf.float32)
    image_rotated = transform_ops.rotate(image_rep, angles)
    np.testing.assert_equal(
        image_rotated.numpy()[:, :, :, 0],
        [
            [
                [0, 3, 8, 9, 0],
                [1, 7, 8, 13, 19],
                [6, 6, 12, 18, 18],
                [5, 11, 16, 17, 23],
                [0, 15, 16, 21, 0],
            ],
            [
                [0, 3, 9, 14, 0],
                [2, 7, 8, 13, 19],
                [1, 6, 12, 18, 23],
                [5, 11, 16, 17, 22],
                [0, 10, 15, 21, 0],
            ],
            [
                [20, 15, 10, 5, 0],
                [21, 16, 11, 6, 1],
                [22, 17, 12, 7, 2],
                [23, 18, 13, 8, 3],
                [24, 19, 14, 9, 4],
            ],
        ],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_compose_rotate(dtype):
    image = tf.constant(
        [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=dtype
    )
    # Rotate counter-clockwise by pi / 2.
    rotation = transform_ops.angles_to_projective_transforms(np.pi / 2, 4, 4)
    # Translate right by 1 (the transformation matrix is always inverted,
    # hence the -1).
    translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0], dtype=tf.float32)
    composed = transform_ops.compose_transforms([rotation, translation])
    image_transformed = transform_ops.transform(image, composed)
    np.testing.assert_equal(
        image_transformed.numpy(),
        [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1]],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_bilinear():
    image = tf.constant(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        tf.float32,
    )
    # The following result matches:
    # >>> scipy.ndimage.rotate(image, 45, order=1, reshape=False)
    # which uses spline interpolation of order 1, equivalent to bilinear
    # interpolation.
    transformed = transform_ops.rotate(image, np.pi / 4.0, interpolation="BILINEAR")
    np.testing.assert_allclose(
        transformed.numpy(),
        [
            [0.000, 0.000, 0.343, 0.000, 0.000],
            [0.000, 0.586, 0.914, 0.586, 0.000],
            [0.343, 0.914, 0.000, 0.914, 0.343],
            [0.000, 0.586, 0.914, 0.586, 0.000],
            [0.000, 0.000, 0.343, 0.000, 0.000],
        ],
        atol=0.001,
    )
    transformed = transform_ops.rotate(image, np.pi / 4.0, interpolation="NEAREST")
    np.testing.assert_allclose(
        transformed.numpy(),
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_bilinear_uint8():
    image = tf.constant(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 255, 255, 255, 0.0],
                [0.0, 255, 0.0, 255, 0.0],
                [0.0, 255, 255, 255, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            np.uint8,
        ),
        tf.uint8,
    )
    # == np.rint((expected image above) * 255)
    transformed = transform_ops.rotate(image, np.pi / 4.0, interpolation="BILINEAR")
    np.testing.assert_equal(
        transformed.numpy(),
        [
            [0.0, 0.0, 87.0, 0.0, 0.0],
            [0.0, 149, 233, 149, 0.0],
            [87.0, 233, 0.0, 233, 87.0],
            [0.0, 149, 233, 149, 0.0],
            [0.0, 0.0, 87.0, 0.0, 0.0],
        ],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_rotate_static_shape():
    image = tf.linalg.diag([1.0, 2.0, 3.0])
    result = transform_ops.rotate(
        image, tf.random.uniform((), -1, 1), interpolation="BILINEAR"
    )
    np.testing.assert_equal(image.get_shape(), result.get_shape())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_unknown_shape():
    fn = tf.function(transform_ops.rotate).get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float32), 0
    )
    for shape in (2, 4), (2, 4, 3), (1, 2, 4, 3):
        image = tf.ones(shape=shape)
        np.testing.assert_equal(image.numpy(), fn(image).numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES - {tf.dtypes.float16})
def test_shear_x(dtype):
    image = np.random.randint(low=0, high=255, size=(4, 4, 3)).astype(
        dtype.as_numpy_dtype
    )
    color = tf.constant([255, 0, 255], tf.int32)
    level = tf.random.uniform(shape=(), minval=0, maxval=1)

    tf_image = tf.constant(image)
    sheared_img = transform_ops.shear_x(tf_image, level, replace=color)
    transform_matrix = transform.AffineTransform(
        np.array([[1, level.numpy(), 0], [0, 1, 0], [0, 0, 1]])
    )
    if dtype == tf.uint8:
        # uint8 can't represent cval=-1, so we use int32 instead
        image = image.astype(np.int32)
    expected_img = transform.warp(
        image, transform_matrix, order=0, cval=-1, preserve_range=True
    )

    mask = np.where(expected_img == -1)
    expected_img[mask[0], mask[1], :] = color

    np.testing.assert_equal(sheared_img.numpy(), expected_img)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES - {tf.dtypes.float16})
def test_shear_y(dtype):
    image = np.random.randint(low=0, high=255, size=(4, 4, 3)).astype(
        dtype.as_numpy_dtype
    )
    color = tf.constant([255, 0, 255], tf.int32)
    level = tf.random.uniform(shape=(), minval=0, maxval=1)

    tf_image = tf.constant(image)
    sheared_img = transform_ops.shear_y(image=tf_image, level=level, replace=color)
    transform_matrix = transform.AffineTransform(
        np.array([[1, 0, 0], [level.numpy(), 1, 0], [0, 0, 1]])
    )
    if dtype == tf.uint8:
        # uint8 can't represent cval=-1, so we use int32 instead
        image = image.astype(np.int32)
    expected_img = transform.warp(
        image, transform_matrix, order=0, cval=-1, preserve_range=True
    )

    mask = np.where(expected_img == -1)
    expected_img[mask[0], mask[1], :] = color

    test_utils.assert_allclose_according_to_type(sheared_img.numpy(), expected_img)
