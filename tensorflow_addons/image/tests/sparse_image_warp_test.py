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

from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.image import sparse_image_warp
from tensorflow_addons.image.sparse_image_warp import _get_boundary_locations
from tensorflow_addons.image.sparse_image_warp import _get_grid_locations
from tensorflow_addons.utils.resource_loader import get_path_to_datafile


def test_zero_shift():
    """Run assert_zero_shift for various hyperparameters."""
    for order in (1, 2):
        for regularization in (0, 0.01):
            for num_boundary_points in (0, 1):
                assert_zero_shift(order, regularization, num_boundary_points)


def assert_zero_shift(order, regularization, num_boundary_points):
    """Check that warping with zero displacements doesn't change the
    image."""
    batch_size = 1
    image_height = 4
    image_width = 4
    channels = 3

    image = np.random.uniform(size=[batch_size, image_height, image_width, channels])

    input_image = tf.constant(np.float32(image))

    control_point_locations = [[1.0, 1.0], [2.0, 2.0], [2.0, 1.0]]
    control_point_locations = tf.constant(
        np.float32(np.expand_dims(control_point_locations, 0))
    )

    control_point_displacements = np.zeros(control_point_locations.shape.as_list())
    control_point_displacements = tf.constant(np.float32(control_point_displacements))

    (warped_image, _) = sparse_image_warp(
        input_image,
        control_point_locations,
        control_point_locations + control_point_displacements,
        interpolation_order=order,
        regularization_weight=regularization,
        num_boundary_points=num_boundary_points,
    )

    np.testing.assert_allclose(warped_image, input_image, rtol=1e-6, atol=1e-6)


def test_get_boundary_locations():
    image_height = 11
    image_width = 11
    num_points_per_edge = 4
    locs = _get_boundary_locations(image_height, image_width, num_points_per_edge)
    num_points = locs.shape[0]
    assert num_points == (4 + 4 * num_points_per_edge)
    locs = [(locs[i, 0], locs[i, 1]) for i in range(num_points)]
    for i in (0, image_height - 1):
        for j in (0, image_width - 1):
            assert (i, j) in locs

        for i in (2, 4, 6, 8):
            for j in (0, image_width - 1):
                assert (i, j) in locs

        for i in (0, image_height - 1):
            for j in (2, 4, 6, 8):
                assert (i, j) in locs


def test_get_grid_locations():
    image_height = 5
    image_width = 3
    grid = _get_grid_locations(image_height, image_width)
    for i in range(image_height):
        for j in range(image_width):
            assert grid[i, j, 0] == i
            assert grid[i, j, 1] == j


def test_move_single_pixel():
    """Run assert_move_single_pixel for various hyperparameters and data
    types."""
    for order in (1, 2):
        for num_boundary_points in (1, 2):
            for type_to_use in (tf.dtypes.float32, tf.dtypes.float64):
                assert_move_single_pixel(order, num_boundary_points, type_to_use)


def assert_move_single_pixel(order, num_boundary_points, type_to_use):
    """Move a single block in a small grid using warping."""
    batch_size = 1
    image_height = 7
    image_width = 7
    channels = 3

    image = np.zeros([batch_size, image_height, image_width, channels])
    image[:, 3, 3, :] = 1.0
    input_image = tf.constant(image, dtype=type_to_use)

    # Place a control point at the one white pixel.
    control_point_locations = [[3.0, 3.0]]
    control_point_locations = tf.constant(
        np.float32(np.expand_dims(control_point_locations, 0)), dtype=type_to_use
    )
    # Shift it one pixel to the right.
    control_point_displacements = [[0.0, 1.0]]
    control_point_displacements = tf.constant(
        np.float32(np.expand_dims(control_point_displacements, 0)), dtype=type_to_use
    )

    (warped_image, flow) = sparse_image_warp(
        input_image,
        control_point_locations,
        control_point_locations + control_point_displacements,
        interpolation_order=order,
        num_boundary_points=num_boundary_points,
    )

    # Check that it moved the pixel correctly.
    np.testing.assert_allclose(
        warped_image[0, 4, 5, :], input_image[0, 4, 4, :], atol=1e-5, rtol=1e-5
    )

    # Test that there is no flow at the corners.
    for i in (0, image_height - 1):
        for j in (0, image_width - 1):
            np.testing.assert_allclose(
                flow[0, i, j, :], np.zeros([2]), atol=1e-5, rtol=1e-5
            )


def load_image(image_file):
    image = tf.image.decode_png(
        tf.io.read_file(image_file), dtype=tf.dtypes.uint8, channels=4
    )[:, :, 0:3]
    return image


def test_smiley_face():
    """Check warping accuracy by comparing to hardcoded warped images."""

    input_file = get_path_to_datafile("image/tests/test_data/Yellow_Smiley_Face.png")
    input_image = load_image(input_file)
    control_points = np.asarray(
        [
            [64, 59],
            [180 - 64, 59],
            [39, 111],
            [180 - 39, 111],
            [90, 143],
            [58, 134],
            [180 - 58, 134],
        ]
    )  # pyformat: disable
    control_point_displacements = np.asarray(
        [
            [-10.5, 10.5],
            [10.5, 10.5],
            [0, 0],
            [0, 0],
            [0, -10],
            [-20, 10.25],
            [10, 10.75],
        ]
    )
    control_points = tf.constant(
        np.expand_dims(np.float32(control_points[:, [1, 0]]), 0)
    )
    control_point_displacements = tf.constant(
        np.expand_dims(np.float32(control_point_displacements[:, [1, 0]]), 0)
    )
    float_image = np.expand_dims(np.float32(input_image) / 255, 0)
    input_image = tf.constant(float_image)

    for interpolation_order in (1, 2, 3):
        for num_boundary_points in (0, 1, 4):
            warped_image, _ = sparse_image_warp(
                input_image,
                control_points,
                control_points + control_point_displacements,
                interpolation_order=interpolation_order,
                num_boundary_points=num_boundary_points,
            )

            warped_image = warped_image
            out_image = np.uint8(warped_image[0, :, :, :] * 255)
            target_file = get_path_to_datafile(
                "image/tests/test_data/Yellow_Smiley_Face_Warp-interp"
                + "-{}-clamp-{}.png".format(interpolation_order, num_boundary_points)
            )

            target_image = load_image(target_file)

            # Check that the target_image and out_image difference is no
            # bigger than 2 (on a scale of 0-255). Due to differences in
            # floating point computation on different devices, the float
            # output in warped_image may get rounded to a different int
            # than that in the saved png file loaded into target_image.
            np.testing.assert_allclose(target_image, out_image, atol=2, rtol=1e-3)


def test_that_backprop_runs():
    """Making sure the gradients can be computed."""
    batch_size = 1
    image_height = 9
    image_width = 12
    image = tf.Variable(
        np.random.uniform(size=[batch_size, image_height, image_width, 3]),
        dtype=tf.float32,
    )
    control_point_locations = [[3.0, 3.0]]
    control_point_locations = tf.constant(
        np.float32(np.expand_dims(control_point_locations, 0))
    )
    control_point_displacements = [[0.25, -0.5]]
    control_point_displacements = tf.constant(
        np.float32(np.expand_dims(control_point_displacements, 0))
    )

    with tf.GradientTape() as t:
        warped_image, _ = sparse_image_warp(
            image,
            control_point_locations,
            control_point_locations + control_point_displacements,
            num_boundary_points=3,
        )

    gradients = t.gradient(warped_image, image).numpy()
    assert np.sum(np.abs(gradients)) != 0


ShapeConfig = namedtuple(
    "ShapeConfig",
    [
        "image",
        "source_control_point_locations",
        "dest_control_point_locations",
        "input",
    ],
)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize(
    "shape",
    [
        ShapeConfig(None, None, None, (1, 9, 12, 3)),
        ShapeConfig(None, [1, 1, 2], [1, 1, 2], (9, 12)),
        ShapeConfig(None, [1, 1, 2], [1, 1, 2], (9, 12, 3)),
        ShapeConfig(None, [1, 1, 2], [1, 1, 2], (1, 9, 12, 3)),
        ShapeConfig([None, 9, 12, 3], [None, 1, 2], [None, 1, 2], (1, 9, 12, 3)),
        ShapeConfig([None, None, None, 3], [None, 1, 2], [None, 1, 2], (1, 9, 12, 3)),
        ShapeConfig(
            [None, None, None, None], [None, 1, 2], [None, 1, 2], (1, 9, 12, 3)
        ),
    ],
)
@pytest.mark.parametrize("interpolation_order", [1, 2, 3])
@pytest.mark.parametrize("num_boundary_points", [1, 2, 3])
def test_partially_or_fully_unknown_shape(
    shape, interpolation_order, num_boundary_points
):
    control_point_locations = np.asarray([3.0, 3.0]).reshape(1, 1, 2).astype(np.float32)
    control_point_displacements = (
        np.asarray([0.25, -0.5]).reshape(1, 1, 2).astype(np.float32)
    )
    fn = tf.function(sparse_image_warp).get_concrete_function(
        image=tf.TensorSpec(shape=shape.image, dtype=tf.float32),
        source_control_point_locations=tf.TensorSpec(
            shape=shape.source_control_point_locations, dtype=tf.float32
        ),
        dest_control_point_locations=tf.TensorSpec(
            shape=shape.dest_control_point_locations, dtype=tf.float32
        ),
        interpolation_order=interpolation_order,
        num_boundary_points=num_boundary_points,
    )
    image = tf.ones(shape=shape.input, dtype=tf.float32)
    expected_output = sparse_image_warp(
        image,
        control_point_locations,
        control_point_locations + control_point_displacements,
        interpolation_order=interpolation_order,
        num_boundary_points=num_boundary_points,
    )
    output = fn(
        image,
        control_point_locations,
        control_point_locations + control_point_displacements,
        interpolation_order=interpolation_order,
        num_boundary_points=num_boundary_points,
        regularization_weight=0.0,
        name="sparse_image_warp",
    )
    np.testing.assert_equal(output[0].numpy(), expected_output[0].numpy())
    np.testing.assert_equal(output[1].numpy(), expected_output[1].numpy())
