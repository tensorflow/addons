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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow_addons.image.python import transform as transform_ops

_DTYPES = set([dtypes.uint8, dtypes.int32, dtypes.int64, dtypes.float16,
               dtypes.float32, dtypes.float64])


class ImageOpsTest(test.TestCase):
    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_compose(self):
        for dtype in _DTYPES:
            image = constant_op.constant(
                [[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                dtype=dtype)
            # Rotate counter-clockwise by pi / 2.
            rotation = transform_ops.angles_to_projective_transforms(np.pi / 2,
                                                                     4, 4)
            # Translate right by 1 (the transformation matrix is always inverted,
            # hence the -1).
            translation = constant_op.constant(
                [1, 0, -1, 0, 1, 0, 0, 0],
                dtype=dtypes.float32)
            composed = transform_ops.compose_transforms(rotation, translation)
            image_transformed = transform_ops.transform(image, composed)
            self.assertAllEqual(
                [[0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1],
                 [0, 1, 1, 1]], image_transformed)

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_extreme_projective_transform(self):
        for dtype in _DTYPES:
            image = constant_op.constant(
                [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
                dtype=dtype)
            transformation = constant_op.constant(
                [1, 0, 0, 0, 1, 0, -1, 0], dtypes.float32)
            image_transformed = transform_ops.transform(image, transformation)
            self.assertAllEqual(
                [[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0],
                 [0, 0, 0, 0]], image_transformed)

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_transform_static_output_shape(self):
        image = constant_op.constant([[1., 2.], [3., 4.]])
        result = transform_ops.transform(
            image,
            random_ops.random_uniform(
                [8], -1, 1),
            output_shape=constant_op.constant([3, 5]))
        self.assertAllEqual([3, 5], result.shape)

    def _test_grad(self, shape_to_test):
        with self.cached_session():
            test_image_shape = shape_to_test
            test_image = np.random.randn(*test_image_shape)
            test_image_tensor = constant_op.constant(test_image,
                                                     shape=test_image_shape)
            test_transform = transform_ops.angles_to_projective_transforms(
                np.pi / 2, 4, 4)

            output_shape = test_image_shape
            output = transform_ops.transform(test_image_tensor, test_transform)
            left_err = gradient_checker.compute_gradient_error(
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
            test_image_tensor = constant_op.constant(test_image,
                                                     shape=test_image_shape)
            test_transform = transform_ops.angles_to_projective_transforms(
                np.pi / 2, 4, 4)

            if len(output_shape) == 2:
                resize_shape = output_shape
            elif len(output_shape) == 3:
                resize_shape = output_shape[0:2]
            elif len(output_shape) == 4:
                resize_shape = output_shape[1:3]
            output = transform_ops.transform(images=test_image_tensor,
                                             transforms=test_transform,
                                             output_shape=resize_shape)
            left_err = gradient_checker.compute_gradient_error(
                test_image_tensor,
                test_image_shape,
                output,
                output_shape,
                x_init_value=test_image)
            self.assertLess(left_err, 1e-10)

    # TODO: switch to TF2 later.
    @tf_test_util.run_deprecated_v1
    def test_grad(self):
        self._test_grad([16, 16])
        self._test_grad([4, 12, 12])
        self._test_grad([3, 4, 12, 12])
        self._test_grad_different_shape([16, 16], [8, 8])
        self._test_grad_different_shape([4, 12, 3], [8, 24, 3])
        self._test_grad_different_shape([3, 4, 12, 3], [3, 8, 24, 3])

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_transform_data_types(self):
        for dtype in _DTYPES:
            image = constant_op.constant([[1, 2], [3, 4]], dtype=dtype)
            with self.test_session(use_gpu=True):
                self.assertAllEqual(
                    np.array([[4, 4], [4, 4]]).astype(dtype.as_numpy_dtype()),
                    transform_ops.transform(image, [1] * 8))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_transform_eager(self):
        image = constant_op.constant([[1., 2.], [3., 4.]])
        self.assertAllEqual(
            np.array([[4, 4], [4, 4]]),
            transform_ops.transform(image, [1] * 8))


if __name__ == "__main__":
    test.main()
