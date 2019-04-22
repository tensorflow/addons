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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.image import transform_ops
from tensorflow_addons.utils import test_utils

_DTYPES = set([
    tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.int64, tf.dtypes.float16,
    tf.dtypes.float32, tf.dtypes.float64
])


class ImageOpsTest(tf.test.TestCase):
    @test_utils.run_in_graph_and_eager_modes
    def test_compose(self):
        for dtype in _DTYPES:
            image = tf.constant(
                [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                dtype=dtype)
            # Rotate counter-clockwise by pi / 2.
            rotation = transform_ops.angles_to_projective_transforms(
                np.pi / 2, 4, 4)
            # Translate right by 1 (the transformation matrix is always inverted,
            # hence the -1).
            translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0],
                                      dtype=tf.dtypes.float32)
            composed = transform_ops.compose_transforms(
                [rotation, translation])
            image_transformed = transform_ops.transform(image, composed)
            self.assertAllEqual(
                [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1]],
                image_transformed)

    @test_utils.run_in_graph_and_eager_modes
    def test_extreme_projective_transform(self):
        for dtype in _DTYPES:
            image = tf.constant(
                [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                dtype=dtype)
            transformation = tf.constant([1, 0, 0, 0, 1, 0, -1, 0],
                                         tf.dtypes.float32)
            image_transformed = transform_ops.transform(image, transformation)
            self.assertAllEqual(
                [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
                image_transformed)

    def test_transform_static_output_shape(self):
        image = tf.constant([[1., 2.], [3., 4.]])
        result = transform_ops.transform(
            image,
            tf.random.uniform([8], -1, 1),
            output_shape=tf.constant([3, 5]))
        self.assertAllEqual([3, 5], result.shape)

    def _test_grad(self, shape_to_test):
        with self.cached_session():
            test_image_shape = shape_to_test
            test_image = np.random.randn(*test_image_shape)
            test_image_tensor = tf.constant(test_image, shape=test_image_shape)
            test_transform = transform_ops.angles_to_projective_transforms(
                np.pi / 2, 4, 4)

            output_shape = test_image_shape
            output = transform_ops.transform(test_image_tensor, test_transform)
            left_err = tf.compat.v1.test.compute_gradient_error(
                test_image_tensor,
                test_image_shape,
                output,
                output_shape,
                x_init_value=test_image)
            self.assertLess(left_err, 1e-10)

    def _test_grad_different_shape(self, input_shape, output_shape):
        with self.cached_session():
            test_image_shape = input_shape
            test_image = np.random.randn(*test_image_shape)
            test_image_tensor = tf.constant(test_image, shape=test_image_shape)
            test_transform = transform_ops.angles_to_projective_transforms(
                np.pi / 2, 4, 4)

            if len(output_shape) == 2:
                resize_shape = output_shape
            elif len(output_shape) == 3:
                resize_shape = output_shape[0:2]
            elif len(output_shape) == 4:
                resize_shape = output_shape[1:3]
            output = transform_ops.transform(
                images=test_image_tensor,
                transforms=test_transform,
                output_shape=resize_shape)
            left_err = tf.compat.v1.test.compute_gradient_error(
                test_image_tensor,
                test_image_shape,
                output,
                output_shape,
                x_init_value=test_image)
            self.assertLess(left_err, 1e-10)

    # TODO: switch to TF2 later.
    @test_utils.run_deprecated_v1
    def test_grad(self):
        self._test_grad([16, 16])
        self._test_grad([4, 12, 12])
        self._test_grad([3, 4, 12, 12])
        self._test_grad_different_shape([16, 16], [8, 8])
        self._test_grad_different_shape([4, 12, 3], [8, 24, 3])
        self._test_grad_different_shape([3, 4, 12, 3], [3, 8, 24, 3])

    @test_utils.run_in_graph_and_eager_modes
    def test_transform_data_types(self):
        for dtype in _DTYPES:
            image = tf.constant([[1, 2], [3, 4]], dtype=dtype)
            self.assertAllEqual(
                np.array([[4, 4], [4, 4]]).astype(dtype.as_numpy_dtype()),
                transform_ops.transform(image, [1] * 8))

    @test_utils.run_in_graph_and_eager_modes
    def test_transform_eager(self):
        image = tf.constant([[1., 2.], [3., 4.]])
        self.assertAllEqual(
            np.array([[4, 4], [4, 4]]), transform_ops.transform(
                image, [1] * 8))


@test_utils.run_all_in_graph_and_eager_modes
class RotateOpTest(tf.test.TestCase):
    def test_zeros(self):
        for dtype in _DTYPES:
            for shape in [(5, 5), (24, 24), (2, 24, 24, 3)]:
                for angle in [0, 1, np.pi / 2.0]:
                    image = tf.zeros(shape, dtype)
                    self.assertAllEqual(
                        transform_ops.rotate(image, angle),
                        np.zeros(shape, dtype.as_numpy_dtype()))

    def test_rotate_even(self):
        for dtype in _DTYPES:
            image = tf.reshape(tf.cast(tf.range(36), dtype), (6, 6))
            image_rep = tf.tile(image[None, :, :, None], [3, 1, 1, 1])
            angles = tf.constant([0.0, np.pi / 4.0, np.pi / 2.0], tf.float32)
            image_rotated = transform_ops.rotate(image_rep, angles)
            # yapf: disable
            self.assertAllEqual(
                image_rotated[:, :, :, 0],
                [[[0, 1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10, 11],
                  [12, 13, 14, 15, 16, 17],
                  [18, 19, 20, 21, 22, 23],
                  [24, 25, 26, 27, 28, 29],
                  [30, 31, 32, 33, 34, 35]],
                 [[0, 3, 4, 11, 17, 0],
                  [2, 3, 9, 16, 23, 23],
                  [1, 8, 15, 21, 22, 29],
                  [6, 13, 20, 21, 27, 34],
                  [12, 18, 19, 26, 33, 33],
                  [0, 18, 24, 31, 32, 0]],
                 [[5, 11, 17, 23, 29, 35],
                  [4, 10, 16, 22, 28, 34],
                  [3, 9, 15, 21, 27, 33],
                  [2, 8, 14, 20, 26, 32],
                  [1, 7, 13, 19, 25, 31],
                  [0, 6, 12, 18, 24, 30]]])
            # yapf: enable

    def test_rotate_odd(self):
        for dtype in _DTYPES:
            image = tf.reshape(tf.cast(tf.range(25), dtype), (5, 5))
            image_rep = tf.tile(image[None, :, :, None], [3, 1, 1, 1])
            angles = tf.constant([np.pi / 4.0, 1.0, -np.pi / 2.0], tf.float32)
            image_rotated = transform_ops.rotate(image_rep, angles)
            # yapf: disable
            self.assertAllEqual(
                image_rotated[:, :, :, 0],
                [[[0, 3, 8, 9, 0],
                  [1, 7, 8, 13, 19],
                  [6, 6, 12, 18, 18],
                  [5, 11, 16, 17, 23],
                  [0, 15, 16, 21, 0]],
                 [[0, 3, 9, 14, 0],
                  [2, 7, 8, 13, 19],
                  [1, 6, 12, 18, 23],
                  [5, 11, 16, 17, 22],
                  [0, 10, 15, 21, 0]],
                 [[20, 15, 10, 5, 0],
                  [21, 16, 11, 6, 1],
                  [22, 17, 12, 7, 2],
                  [23, 18, 13, 8, 3],
                  [24, 19, 14, 9, 4]]])
            # yapf: enable

    def test_compose_rotate(self):
        for dtype in _DTYPES:
            image = tf.constant(
                [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                dtype=dtype)
            # Rotate counter-clockwise by pi / 2.
            rotation = transform_ops.angles_to_projective_transforms(
                np.pi / 2, 4, 4)
            # Translate right by 1 (the transformation matrix is always inverted,
            # hence the -1).
            translation = tf.constant([1, 0, -1, 0, 1, 0, 0, 0],
                                      dtype=tf.float32)
            composed = transform_ops.compose_transforms(
                [rotation, translation])
            image_transformed = transform_ops.transform(image, composed)
            self.assertAllEqual(
                image_transformed,
                [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 1, 1]])

    def test_bilinear(self):
        image = tf.constant(
            # yapf: disable
            [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0],
             [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            # yapf: enable
            tf.float32)
        # The following result matches:
        # >>> scipy.ndimage.rotate(image, 45, order=1, reshape=False)
        # which uses spline interpolation of order 1, equivalent to bilinear
        # interpolation.
        self.assertAllClose(
            transform_ops.rotate(image, np.pi / 4.0, interpolation="BILINEAR"),
            # yapf: disable
            [[0.000, 0.000, 0.343, 0.000, 0.000],
             [0.000, 0.586, 0.914, 0.586, 0.000],
             [0.343, 0.914, 0.000, 0.914, 0.343],
             [0.000, 0.586, 0.914, 0.586, 0.000],
             [0.000, 0.000, 0.343, 0.000, 0.000]],
            # yapf: enable
            atol=0.001)
        # yapf: disable
        self.assertAllClose(
            transform_ops.rotate(
                image, np.pi / 4.0, interpolation="NEAREST"),
            [[0, 0, 1, 0, 0],
             [0, 1, 1, 1, 0],
             [1, 1, 0, 1, 1],
             [0, 1, 1, 1, 0],
             [0, 0, 1, 0, 0]])
        # yapf: enable

    def test_bilinear_uint8(self):
        # yapf: disable
        image = tf.constant(
            np.asarray(
                [[0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 255, 255, 255, 0.0],
                 [0.0, 255, 0.0, 255, 0.0],
                 [0.0, 255, 255, 255, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0]],
                np.uint8),
            tf.uint8)
        # yapf: enable
        # == np.rint((expected image above) * 255)
        # yapf: disable
        self.assertAllEqual(
            transform_ops.rotate(image, np.pi / 4.0, interpolation="BILINEAR"),
            [[0.0, 0.0, 87., 0.0, 0.0], [0.0, 149, 233, 149, 0.0],
             [87., 233, 0.0, 233, 87.], [0.0, 149, 233, 149, 0.0],
             [0.0, 0.0, 87., 0.0, 0.0]])
        # yapf: enable

    def test_rotate_static_shape(self):
        image = tf.linalg.diag([1., 2., 3.])
        result = transform_ops.rotate(
            image, tf.random.uniform((), -1, 1), interpolation="BILINEAR")
        self.assertEqual(image.get_shape(), result.get_shape())


if __name__ == "__main__":
    tf.test.main()
