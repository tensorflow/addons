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
"""Tests for dense_image_warp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

from tensorflow_addons.image import dense_image_warp
from tensorflow_addons.image import interpolate_bilinear
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class InterpolateBilinearTest(tf.test.TestCase):
    def test_interpolate_small_grid_ij(self):
        grid = tf.constant(
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]],
            shape=[1, 4, 3, 1])
        query_points = tf.constant(
            [[0., 0.], [1., 0.], [2., 0.5], [1.5, 1.5], [3., 2.]],
            shape=[1, 5, 2])
        expected_results = np.reshape(
            np.array([0., 3., 6.5, 6., 11.]), [1, 5, 1])

        interp = interpolate_bilinear(grid, query_points)

        self.assertAllClose(expected_results, interp)

    def test_interpolate_small_grid_xy(self):
        grid = tf.constant(
            [[0., 1., 2.], [3., 4., 5.], [6., 7., 8.], [9., 10., 11.]],
            shape=[1, 4, 3, 1])
        query_points = tf.constant(
            [[0., 0.], [0., 1.], [0.5, 2.0], [1.5, 1.5], [2., 3.]],
            shape=[1, 5, 2])
        expected_results = np.reshape(
            np.array([0., 3., 6.5, 6., 11.]), [1, 5, 1])

        interp = interpolate_bilinear(grid, query_points, indexing="xy")

        self.assertAllClose(expected_results, interp)

    def test_interpolate_small_grid_batched(self):
        grid = tf.constant([[[0., 1.], [3., 4.]], [[5., 6.], [7., 8.]]],
                           shape=[2, 2, 2, 1])
        query_points = tf.constant([[[0., 0.], [1., 0.], [0.5, 0.5]],
                                    [[0.5, 0.], [1., 0.], [1., 1.]]])
        expected_results = np.reshape(
            np.array([[0., 3., 2.], [6., 7., 8.]]), [2, 3, 1])

        interp = interpolate_bilinear(grid, query_points)

        self.assertAllClose(expected_results, interp)

    def test_unknown_shape(self):
        query_points = tf.constant(
            [[0., 0.], [0., 1.], [0.5, 2.0], [1.5, 1.5]], shape=[1, 4, 2])
        fn = interpolate_bilinear.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.float32))
        for shape in (2, 4, 3, 6), (6, 2, 4, 3), (1, 2, 4, 3):
            image = tf.ones(shape=shape)
            res = fn(image, query_points)
            self.assertAllEqual(res.shape, (shape[0], 4, shape[3]))


@test_utils.run_all_in_graph_and_eager_modes
class DenseImageWarpTest(tf.test.TestCase):
    def _get_random_image_and_flows(self, shape, image_type, flow_type):
        batch_size, height, width, num_channels = shape
        image_shape = [batch_size, height, width, num_channels]
        image = np.random.normal(size=image_shape)
        flow_shape = [batch_size, height, width, 2]
        flows = np.random.normal(size=flow_shape) * 3
        return image.astype(image_type), flows.astype(flow_type)

    def _assert_correct_interpolation_value(self,
                                            image,
                                            flows,
                                            pred_interpolation,
                                            batch_index,
                                            y_index,
                                            x_index,
                                            low_precision=False):
        """Assert that the tf interpolation matches hand-computed value."""
        height = image.shape[1]
        width = image.shape[2]
        displacement = flows[batch_index, y_index, x_index, :]
        float_y = y_index - displacement[0]
        float_x = x_index - displacement[1]
        floor_y = max(min(height - 2, math.floor(float_y)), 0)
        floor_x = max(min(width - 2, math.floor(float_x)), 0)
        ceil_y = floor_y + 1
        ceil_x = floor_x + 1

        alpha_y = min(max(0.0, float_y - floor_y), 1.0)
        alpha_x = min(max(0.0, float_x - floor_x), 1.0)

        floor_y = int(floor_y)
        floor_x = int(floor_x)
        ceil_y = int(ceil_y)
        ceil_x = int(ceil_x)

        top_left = image[batch_index, floor_y, floor_x, :]
        top_right = image[batch_index, floor_y, ceil_x, :]
        bottom_left = image[batch_index, ceil_y, floor_x, :]
        bottom_right = image[batch_index, ceil_y, ceil_x, :]

        interp_top = alpha_x * (top_right - top_left) + top_left
        interp_bottom = alpha_x * (bottom_right - bottom_left) + bottom_left
        interp = alpha_y * (interp_bottom - interp_top) + interp_top
        atol = 1e-6
        rtol = 1e-6
        if low_precision:
            atol = 1e-2
            rtol = 1e-3
        self.assertAllClose(
            interp,
            pred_interpolation[batch_index, y_index, x_index, :],
            atol=atol,
            rtol=rtol)

    def _check_zero_flow_correctness(self, shape, image_type, flow_type):
        """Assert using zero flows doesn't change the input image."""
        rand_image, rand_flows = self._get_random_image_and_flows(
            shape, image_type, flow_type)
        rand_flows *= 0

        interp = dense_image_warp(
            image=tf.convert_to_tensor(rand_image),
            flow=tf.convert_to_tensor(rand_flows))

        self.assertAllClose(rand_image, interp)

    def test_zero_flows(self):
        """Apply _check_zero_flow_correctness() for a few sizes and types."""
        shapes_to_try = [[3, 4, 5, 6], [1, 2, 2, 1]]
        for shape in shapes_to_try:
            self._check_zero_flow_correctness(
                shape, image_type="float32", flow_type="float32")

    def _check_interpolation_correctness(self,
                                         shape,
                                         image_type,
                                         flow_type,
                                         call_with_unknown_shapes=False,
                                         num_probes=5):
        """Interpolate, and then assert correctness for a few query
        locations."""
        low_precision = image_type == "float16" or flow_type == "float16"
        rand_image, rand_flows = self._get_random_image_and_flows(
            shape, image_type, flow_type)

        if call_with_unknown_shapes:
            fn = dense_image_warp.get_concrete_function(
                tf.TensorSpec(shape=None, dtype=image_type),
                tf.TensorSpec(shape=None, dtype=flow_type))
            interp = fn(
                image=tf.convert_to_tensor(rand_image),
                flow=tf.convert_to_tensor(rand_flows))
        else:
            interp = dense_image_warp(
                image=tf.convert_to_tensor(rand_image),
                flow=tf.convert_to_tensor(rand_flows))

        for _ in range(num_probes):
            batch_index = np.random.randint(0, shape[0])
            y_index = np.random.randint(0, shape[1])
            x_index = np.random.randint(0, shape[2])

            self._assert_correct_interpolation_value(
                rand_image,
                rand_flows,
                interp,
                batch_index,
                y_index,
                x_index,
                low_precision=low_precision)

    def test_interpolation(self):
        """Apply _check_interpolation_correctness() for a few sizes and
        types."""
        shapes_to_try = [[3, 4, 5, 6], [1, 5, 5, 3], [1, 2, 2, 1]]
        for im_type in ["float32", "float64", "float16"]:
            for flow_type in ["float32", "float64", "float16"]:
                for shape in shapes_to_try:
                    self._check_interpolation_correctness(
                        shape, im_type, flow_type)

    def test_unknown_shapes(self):
        """Apply _check_interpolation_correctness() for a few sizes and check
        for tf.Dataset compatibility."""
        shapes_to_try = [[3, 4, 5, 6], [1, 5, 5, 3], [1, 2, 2, 1]]
        for shape in shapes_to_try:
            self._check_interpolation_correctness(shape, "float32", "float32",
                                                  True)

    def test_gradients_exist(self):
        """Check that backprop can run.

        The correctness of the gradients is assumed, since the forward
        propagation is tested to be correct and we only use built-in tf
        ops. However, we perform a simple test to make sure that
        backprop can actually run. We treat the flows as a tf.Variable
        and optimize them to minimize the difference between the
        interpolated image and the input image.
        """
        batch_size, height, width, num_channels = [4, 5, 6, 7]
        image_shape = [batch_size, height, width, num_channels]
        image = tf.random.normal(image_shape)
        flow_shape = [batch_size, height, width, 2]
        flows = tf.Variable(
            tf.random.normal(shape=flow_shape) * 0.25, dtype=tf.float32)

        def loss():
            interp = dense_image_warp(image, flows)
            return tf.math.reduce_mean(tf.math.square(interp - image))

        optimizer = tf.keras.optimizers.Adam(1.0)
        minimize_op = optimizer.minimize(loss, var_list=[flows])

        self.evaluate(tf.compat.v1.global_variables_initializer())

        for _ in range(10):
            self.evaluate(minimize_op)

    def test_size_exception(self):
        """Make sure it throws an exception for images that are too small."""
        shape = [1, 2, 1, 1]
        errors = (ValueError, tf.errors.InvalidArgumentError)
        with self.assertRaisesRegexp(errors, "Grid width must be at least 2."):
            self._check_interpolation_correctness(shape, "float32", "float32")


if __name__ == "__main__":
    tf.test.main()
