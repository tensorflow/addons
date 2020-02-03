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
"""Tests for sparse_image_warp."""

import numpy as np
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
from tensorflow_addons.image.sparse_image_warp import _get_boundary_locations
from tensorflow_addons.image.sparse_image_warp import _get_grid_locations
from tensorflow_addons.utils import test_utils
from tensorflow_addons.utils.resource_loader import get_path_to_datafile


@test_utils.run_all_in_graph_and_eager_modes
class SparseImageWarpTest(tf.test.TestCase):
    def setUp(self):
        np.random.seed(0)

    def testGetBoundaryLocations(self):
        image_height = 11
        image_width = 11
        num_points_per_edge = 4
        locs = _get_boundary_locations(image_height, image_width,
                                       num_points_per_edge)
        num_points = locs.shape[0]
        self.assertEqual(num_points, 4 + 4 * num_points_per_edge)
        locs = [(locs[i, 0], locs[i, 1]) for i in range(num_points)]
        for i in (0, image_height - 1):
            for j in (0, image_width - 1):
                self.assertIn((i, j), locs,
                              '{},{} not in the locations'.format(i, j))

            for i in (2, 4, 6, 8):
                for j in (0, image_width - 1):
                    self.assertIn((i, j), locs,
                                  '{},{} not in the locations'.format(i, j))

            for i in (0, image_height - 1):
                for j in (2, 4, 6, 8):
                    self.assertIn((i, j), locs,
                                  '{},{} not in the locations'.format(i, j))

    def testGetGridLocations(self):
        image_height = 5
        image_width = 3
        grid = _get_grid_locations(image_height, image_width)
        for i in range(image_height):
            for j in range(image_width):
                self.assertEqual(grid[i, j, 0], i)
                self.assertEqual(grid[i, j, 1], j)

    def testZeroShift(self):
        """Run assertZeroShift for various hyperparameters."""
        for order in (1, 2):
            for regularization in (0, 0.01):
                for num_boundary_points in (0, 1):
                    self.assertZeroShift(order, regularization,
                                         num_boundary_points)

    def assertZeroShift(self, order, regularization, num_boundary_points):
        """Check that warping with zero displacements doesn't change the
        image."""
        batch_size = 1
        image_height = 4
        image_width = 4
        channels = 3

        image = np.random.uniform(
            size=[batch_size, image_height, image_width, channels])

        input_image = tf.constant(np.float32(image))

        control_point_locations = [[1., 1.], [2., 2.], [2., 1.]]
        control_point_locations = tf.constant(
            np.float32(np.expand_dims(control_point_locations, 0)))

        control_point_displacements = np.zeros(
            control_point_locations.shape.as_list())
        control_point_displacements = tf.constant(
            np.float32(control_point_displacements))

        (warped_image, flow) = sparse_image_warp(
            input_image,
            control_point_locations,
            control_point_locations + control_point_displacements,
            interpolation_order=order,
            regularization_weight=regularization,
            num_boundary_points=num_boundary_points)

        warped_image, input_image = self.evaluate([warped_image, input_image])
        self.assertAllClose(warped_image, input_image)

    def testMoveSinglePixel(self):
        """Run assertMoveSinglePixel for various hyperparameters and data
        types."""
        for order in (1, 2):
            for num_boundary_points in (1, 2):
                for type_to_use in (tf.dtypes.float32, tf.dtypes.float64):
                    self.assertMoveSinglePixel(order, num_boundary_points,
                                               type_to_use)

    def assertMoveSinglePixel(self, order, num_boundary_points, type_to_use):
        """Move a single block in a small grid using warping."""
        batch_size = 1
        image_height = 7
        image_width = 7
        channels = 3

        image = np.zeros([batch_size, image_height, image_width, channels])
        image[:, 3, 3, :] = 1.0
        input_image = tf.constant(image, dtype=type_to_use)

        # Place a control point at the one white pixel.
        control_point_locations = [[3., 3.]]
        control_point_locations = tf.constant(
            np.float32(np.expand_dims(control_point_locations, 0)),
            dtype=type_to_use)
        # Shift it one pixel to the right.
        control_point_displacements = [[0., 1.0]]
        control_point_displacements = tf.constant(
            np.float32(np.expand_dims(control_point_displacements, 0)),
            dtype=type_to_use)

        (warped_image, flow) = sparse_image_warp(
            input_image,
            control_point_locations,
            control_point_locations + control_point_displacements,
            interpolation_order=order,
            num_boundary_points=num_boundary_points)

        warped_image, input_image, flow = self.evaluate(
            [warped_image, input_image, flow])
        # Check that it moved the pixel correctly.
        self.assertAllClose(
            warped_image[0, 4, 5, :],
            input_image[0, 4, 4, :],
            atol=1e-5,
            rtol=1e-5)

        # Test that there is no flow at the corners.
        for i in (0, image_height - 1):
            for j in (0, image_width - 1):
                self.assertAllClose(
                    flow[0, i, j, :], np.zeros([2]), atol=1e-5, rtol=1e-5)

    def load_image(self, image_file):
        image = tf.image.decode_png(
            tf.io.read_file(image_file), dtype=tf.dtypes.uint8,
            channels=4)[:, :, 0:3]
        return self.evaluate(image)

    def testSmileyFace(self):
        """Check warping accuracy by comparing to hardcoded warped images."""

        input_file = get_path_to_datafile(
            "image/test_data/Yellow_Smiley_Face.png")
        input_image = self.load_image(input_file)
        control_points = np.asarray([[64, 59], [180 - 64, 59], [39, 111],
                                     [180 - 39, 111], [90, 143], [58, 134],
                                     [180 - 58, 134]])  # pyformat: disable
        control_point_displacements = np.asarray([[-10.5, 10.5], [10.5, 10.5],
                                                  [0, 0], [0, 0], [0, -10],
                                                  [-20, 10.25], [10, 10.75]])
        control_points = tf.constant(
            np.expand_dims(np.float32(control_points[:, [1, 0]]), 0))
        control_point_displacements = tf.constant(
            np.expand_dims(
                np.float32(control_point_displacements[:, [1, 0]]), 0))
        float_image = np.expand_dims(np.float32(input_image) / 255, 0)
        input_image = tf.constant(float_image)

        for interpolation_order in (1, 2, 3):
            for num_boundary_points in (0, 1, 4):
                warped_image, _ = sparse_image_warp(
                    input_image,
                    control_points,
                    control_points + control_point_displacements,
                    interpolation_order=interpolation_order,
                    num_boundary_points=num_boundary_points)

                warped_image = self.evaluate(warped_image)
                out_image = np.uint8(warped_image[0, :, :, :] * 255)
                target_file = get_path_to_datafile(
                    "image/test_data/Yellow_Smiley_Face_Warp-interp" +
                    "-{}-clamp-{}.png".format(interpolation_order,
                                              num_boundary_points))

                target_image = self.load_image(target_file)

                # Check that the target_image and out_image difference is no
                # bigger than 2 (on a scale of 0-255). Due to differences in
                # floating point computation on different devices, the float
                # output in warped_image may get rounded to a different int
                # than that in the saved png file loaded into target_image.
                self.assertAllClose(target_image, out_image, atol=2, rtol=1e-3)

    def testThatBackpropRuns(self):
        """Run optimization to ensure that gradients can be computed."""
        batch_size = 1
        image_height = 9
        image_width = 12
        image = tf.Variable(
            np.random.uniform(size=[batch_size, image_height, image_width, 3]),
            dtype=tf.float32)
        control_point_locations = [[3., 3.]]
        control_point_locations = tf.constant(
            np.float32(np.expand_dims(control_point_locations, 0)))
        control_point_displacements = [[0.25, -0.5]]
        control_point_displacements = tf.constant(
            np.float32(np.expand_dims(control_point_displacements, 0)))

        def loss_fn():
            warped_image, _ = sparse_image_warp(
                image,
                control_point_locations,
                control_point_locations + control_point_displacements,
                num_boundary_points=3)
            loss = tf.reduce_mean(tf.abs(warped_image - image))
            return loss

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=0.001, momentum=0.9, clipnorm=1.0)
        opt_op = optimizer.minimize(loss_fn, [image])

        self.evaluate(tf.compat.v1.global_variables_initializer())
        for _ in range(5):
            self.evaluate(opt_op)


if __name__ == "__main__":
    tf.test.main()
