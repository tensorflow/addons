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

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.image import mean_filter2d
from tensorflow_addons.image import median_filter2d
from tensorflow_addons.image import gaussian_filter2d
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian

_dtypes_to_test = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}

_image_shapes_to_test = [
    (3, 3, 1),
    (3, 3, 3),
    (1, 3, 3, 1),
    (1, 3, 3, 3),
    (2, 3, 3, 1),
    (2, 3, 3, 3),
]


def tile_image(plane, image_shape):
    """Tile a 2-D image `plane` into 3-D or 4-D as per `image_shape`."""
    assert 3 <= len(image_shape) <= 4
    plane = tf.convert_to_tensor(plane)
    plane = tf.expand_dims(plane, -1)
    channels = image_shape[-1]
    image = tf.tile(plane, (1, 1, channels))

    if len(image_shape) == 4:
        batch_size = image_shape[0]
        image = tf.expand_dims(image, 0)
        image = tf.tile(image, (batch_size, 1, 1, 1))

    return image


def setup_values(
    filter2d_fn, image_shape, filter_shape, padding, constant_values, dtype
):
    assert 3 <= len(image_shape) <= 4
    height, width = image_shape[-3], image_shape[-2]
    plane = tf.constant(
        [x for x in range(1, height * width + 1)], shape=(height, width), dtype=dtype,
    )
    image = tile_image(plane, image_shape=image_shape)

    result = filter2d_fn(
        image,
        filter_shape=filter_shape,
        padding=padding,
        constant_values=constant_values,
    )

    return result


def verify_values(
    filter2d_fn, image_shape, filter_shape, padding, constant_values, expected_plane
):
    expected_output = tile_image(expected_plane, image_shape)
    for dtype in _dtypes_to_test:
        result = setup_values(
            filter2d_fn, image_shape, filter_shape, padding, constant_values, dtype
        )
        np.testing.assert_allclose(
            result.numpy(),
            tf.dtypes.cast(expected_output, dtype).numpy(),
            rtol=1e-02,
            atol=1e-02,
        )

    def setUp(self):
        self._filter2d_fn = mean_filter2d
        super().setUp()


@pytest.mark.parametrize("image_shape", [(1,), (16, 28, 28, 1, 1)])
def test_invalid_image_mean(image_shape):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        image = tf.ones(shape=image_shape)
        mean_filter2d(image)


@pytest.mark.parametrize("filter_shape", [(3, 3, 3), (3, None, 3)])
def test_invalid_filter_shape_mean(filter_shape):
    image = tf.ones(shape=(1, 28, 28, 1))

    with pytest.raises(ValueError):
        mean_filter2d(image, filter_shape=filter_shape)

    filter_shape = None
    with pytest.raises(TypeError):
        mean_filter2d(image, filter_shape=filter_shape)


def test_invalid_padding_mean():
    image = tf.ones(shape=(1, 28, 28, 1))

    with pytest.raises(ValueError):
        mean_filter2d(image, padding="TEST")


def test_none_channels_mean():
    # 3-D image
    fn = mean_filter2d.get_concrete_function(
        tf.TensorSpec(dtype=tf.dtypes.float32, shape=(3, 3, None))
    )
    fn(tf.ones(shape=(3, 3, 1)))
    fn(tf.ones(shape=(3, 3, 3)))

    # 4-D image
    fn = mean_filter2d.get_concrete_function(
        tf.TensorSpec(dtype=tf.dtypes.float32, shape=(1, 3, 3, None))
    )
    fn(tf.ones(shape=(1, 3, 3, 1)))
    fn(tf.ones(shape=(1, 3, 3, 3)))


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3), (1, 3, 3, 3)])
def test_unknown_shape_mean(shape):
    fn = mean_filter2d.get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.dtypes.float32),
        padding="CONSTANT",
        constant_values=1.0,
    )

    image = tf.ones(shape=shape)
    np.testing.assert_equal(image.numpy(), fn(image).numpy())


@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_reflect_padding_with_3x3_filter_mean(image_shape):
    expected_plane = tf.constant(
        [
            [3.6666667, 4.0, 4.3333335],
            [4.6666665, 5.0, 5.3333335],
            [5.6666665, 6.0, 6.3333335],
        ]
    )

    verify_values(
        mean_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="REFLECT",
        constant_values=0,
        expected_plane=expected_plane,
    )


@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_reflect_padding_with_4x4_filter_mean(image_shape):
    expected_plane = tf.constant([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0],])

    verify_values(
        mean_filter2d,
        image_shape=image_shape,
        filter_shape=(4, 4),
        padding="REFLECT",
        constant_values=0,
        expected_plane=expected_plane,
    )


@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_constant_padding_with_3x3_filter_mean(image_shape):
    expected_plane = tf.constant(
        [
            [1.3333334, 2.3333333, 1.7777778],
            [3.0, 5.0, 3.6666667],
            [2.6666667, 4.3333335, 3.1111112],
        ]
    )

    verify_values(
        mean_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="CONSTANT",
        constant_values=0,
        expected_plane=expected_plane,
    )

    expected_plane = tf.constant(
        [
            [1.8888888, 2.6666667, 2.3333333],
            [3.3333333, 5.0, 4.0],
            [3.2222223, 4.6666665, 3.6666667],
        ]
    )

    verify_values(
        mean_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="CONSTANT",
        constant_values=1,
        expected_plane=expected_plane,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_symmetric_padding_with_3x3_filter_mean(image_shape):
    expected_plane = tf.constant(
        [
            [2.3333333, 3.0, 3.6666667],
            [4.3333335, 5.0, 5.6666665],
            [6.3333335, 7.0, 7.6666665],
        ]
    )

    verify_values(
        mean_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="SYMMETRIC",
        constant_values=0,
        expected_plane=expected_plane,
    )


@pytest.mark.parametrize("image_shape", [(1,), (16, 28, 28, 1, 1)])
def test_invalid_image_median(image_shape):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        image = tf.ones(shape=image_shape)
        median_filter2d(image)


@pytest.mark.parametrize("filter_shape", [(3, 3, 3), (3, None, 3)])
def test_invalid_filter_shape_median(filter_shape):
    image = tf.ones(shape=(1, 28, 28, 1))

    with pytest.raises(ValueError):
        median_filter2d(image, filter_shape=filter_shape)

    filter_shape = None
    with pytest.raises(TypeError):
        mean_filter2d(image, filter_shape=filter_shape)


def test_invalid_padding_median():
    image = tf.ones(shape=(1, 28, 28, 1))

    with pytest.raises(ValueError):
        median_filter2d(image, padding="TEST")


def test_none_channels_median():
    # 3-D image
    fn = median_filter2d.get_concrete_function(
        tf.TensorSpec(dtype=tf.dtypes.float32, shape=(3, 3, None))
    )
    fn(tf.ones(shape=(3, 3, 1)))
    fn(tf.ones(shape=(3, 3, 3)))

    # 4-D image
    fn = median_filter2d.get_concrete_function(
        tf.TensorSpec(dtype=tf.dtypes.float32, shape=(1, 3, 3, None))
    )
    fn(tf.ones(shape=(1, 3, 3, 1)))
    fn(tf.ones(shape=(1, 3, 3, 3)))


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3), (1, 3, 3, 3)])
def test_unknown_shape_median(shape):
    fn = median_filter2d.get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.dtypes.float32),
        padding="CONSTANT",
        constant_values=1.0,
    )

    image = tf.ones(shape=shape)
    np.testing.assert_equal(image.numpy(), fn(image).numpy())


@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_reflect_padding_with_3x3_filter_median(image_shape):
    expected_plane = tf.constant([[4, 4, 5], [5, 5, 5], [5, 6, 6]])

    verify_values(
        median_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="REFLECT",
        constant_values=0,
        expected_plane=expected_plane,
    )


@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_reflect_padding_with_4x4_filter_median(image_shape):
    expected_plane = tf.constant([[5, 5, 5], [5, 5, 5], [5, 5, 5]])

    verify_values(
        median_filter2d,
        image_shape=image_shape,
        filter_shape=(4, 4),
        padding="REFLECT",
        constant_values=0,
        expected_plane=expected_plane,
    )


@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_constant_padding_with_3x3_filter(image_shape):
    expected_plane = tf.constant([[0, 2, 0], [2, 5, 3], [0, 5, 0]])

    verify_values(
        median_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="CONSTANT",
        constant_values=0,
        expected_plane=expected_plane,
    )

    expected_plane = tf.constant([[1, 2, 1], [2, 5, 3], [1, 5, 1]])

    verify_values(
        median_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="CONSTANT",
        constant_values=1,
        expected_plane=expected_plane,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("image_shape", _image_shapes_to_test)
def test_symmetric_padding_with_3x3_filter_median(image_shape):
    expected_plane = tf.constant([[2, 3, 3], [4, 5, 6], [7, 7, 8]])

    verify_values(
        median_filter2d,
        image_shape=image_shape,
        filter_shape=(3, 3),
        padding="SYMMETRIC",
        constant_values=0,
        expected_plane=expected_plane,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_gaussian_filter2d_constant():
    test_image_tf = tf.random.uniform(
        [1, 40, 40, 1], minval=0, maxval=255, dtype=tf.float64
    )
    gb = gaussian_filter2d(test_image_tf, 5, 1, padding="CONSTANT")
    gb = gb.numpy()
    gb1 = np.resize(gb, (40, 40))
    test_image_cv = test_image_tf.numpy()
    test_image_cv = np.resize(test_image_cv, [40, 40])
    gb2 = gaussian_filter(test_image_cv, 1, truncate=4.6, mode="constant")
    np.testing.assert_allclose(gb2, gb1, 0.06)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_gaussian_filter2d_reflect():
    test_image_tf = tf.random.uniform(
        [1, 40, 40, 1], minval=0, maxval=255, dtype=tf.int32
    )
    gb = gaussian_filter2d(test_image_tf, 5, 1, padding="REFLECT")
    gb = gb.numpy()
    gb1 = np.resize(gb, (40, 40))
    test_image_cv = test_image_tf.numpy()
    test_image_cv = np.resize(test_image_cv, [40, 40])
    gb2 = gaussian_filter(test_image_cv, 1, truncate=4.6, mode="mirror")
    np.testing.assert_allclose(gb2, gb1, 0.06)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_gaussian_filter2d_symmetric():
    test_image_tf = tf.random.uniform(
        [1, 40, 40, 1], minval=0, maxval=255, dtype=tf.float64
    )
    gb = gaussian_filter2d(test_image_tf, (5, 5), 1, padding="SYMMETRIC")
    gb = gb.numpy()
    gb1 = np.resize(gb, (40, 40))
    test_image_cv = test_image_tf.numpy()
    test_image_cv = np.resize(test_image_cv, [40, 40])
    gb2 = gaussian_filter(test_image_cv, 1, truncate=4.6, mode="reflect")
    np.testing.assert_allclose(gb2, gb1, 0.06)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("image_shape", [[2, 5, 5, 3]])
def test_gaussian_filter2d_batch(image_shape):
    test_image_tf = tf.random.uniform(
        [1, 40, 40, 1], minval=0, maxval=255, dtype=tf.float32
    )
    gb = gaussian_filter2d(test_image_tf, 5, 1, padding="SYMMETRIC")
    gb = gb.numpy()
    gb1 = np.resize(gb, (40, 40))
    test_image_cv = test_image_tf.numpy()
    test_image_cv = np.resize(test_image_cv, [40, 40])
    gb2 = gaussian(test_image_cv, 1, truncate=4.6, mode="reflect")
    np.testing.assert_allclose(gb2, gb1, 0.06)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_gaussian_filter2d_channels():
    test_image_tf = tf.constant(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 2.0, 0.0],
                    [4.0, 4.0, 0.0],
                    [6.0, 6.0, 0.0],
                    [8.0, 8.0, 0.0],
                ],
                [
                    [10.0, 10.0, 0.0],
                    [12.0, 12.0, 0.0],
                    [14.0, 14.0, 0.0],
                    [16.0, 16.0, 0.0],
                    [18.0, 18.0, 0.0],
                ],
                [
                    [20.0, 20.0, 0.0],
                    [22.0, 22.0, 0.0],
                    [24.0, 24.0, 0.0],
                    [26.0, 26.0, 0.0],
                    [28.0, 28.0, 0.0],
                ],
                [
                    [30.0, 30.0, 0.0],
                    [32.0, 32.0, 0.0],
                    [34.0, 34.0, 0.0],
                    [36.0, 36.0, 0.0],
                    [38.0, 38.0, 0.0],
                ],
                [
                    [40.0, 40.0, 0.0],
                    [42.0, 42.0, 0.0],
                    [44.0, 44.0, 0.0],
                    [46.0, 46.0, 0.0],
                    [48.0, 48.0, 0.0],
                ],
            ]
        ],
        dtype=tf.float32,
    )
    gb = gaussian_filter2d(test_image_tf, 5, 1, padding="SYMMETRIC", name="gaussian")
    gb = gb.numpy()
    gb1 = np.resize(gb, (5, 5, 3))
    test_image_cv = test_image_tf.numpy()
    test_image_cv = np.resize(test_image_cv, [5, 5, 3])
    gb2 = gaussian(test_image_cv, sigma=1, mode="reflect", multichannel=True)
    np.testing.assert_allclose(gb2, gb1, 0.06)
