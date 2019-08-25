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
"""Image warping using sparse flow defined at control points."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.image import dense_image_warp
from tensorflow_addons.image import interpolate_spline


def _get_grid_locations(image_height, image_width):
    """Wrapper for tf.meshgrid."""

    y_range = tf.linspace(0.0, tf.cast(image_height, tf.float32) - 1, image_height)
    x_range = tf.linspace(0.0, tf.cast(image_width, tf.float32) - 1, image_width)
    y_grid, x_grid = tf.meshgrid(y_range, x_range, indexing='ij')
    return tf.stack((y_grid, x_grid), -1)


def _expand_to_minibatch(array, batch_size):
    """Tile arbitrarily-sized array to include new batch dimension."""
    batch_size = tf.expand_dims(batch_size, 0)
    array_ones = tf.ones((tf.rank(array)), dtype=tf.dtypes.int32)
    tiles = tf.concat([batch_size, array_ones], axis=0)
    return tf.tile(tf.expand_dims(array, 0), tiles)


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
    """Compute evenly-spaced indices along edge of image."""
    image_height = tf.cast(image_height, tf.float32)
    image_width = tf.cast(image_width, tf.float32)
    y_range = tf.linspace(0.0, image_height - 1, num_points_per_edge + 2)
    x_range = tf.linspace(0.0, image_width - 1, num_points_per_edge + 2)
    ys, xs = tf.meshgrid(y_range, x_range, indexing='ij')
    is_boundary = tf.logical_or(
        tf.logical_or(tf.equal(xs, 0), tf.equal(xs, image_width - 1)),
        tf.logical_or(tf.equal(ys, 0), tf.equal(ys, image_height - 1)))
    return tf.stack([tf.boolean_mask(ys, is_boundary),
                     tf.boolean_mask(xs, is_boundary)], axis=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):
    """Add control points for zero-flow boundary conditions.

    Augment the set of control points with extra points on the
    boundary of the image that have zero flow.

    Args:
      control_point_locations: input control points
      control_point_flows: their flows
      image_height: image height
      image_width: image width
      boundary_points_per_edge: number of points to add in the middle of each
                            edge (not including the corners).
                            The total number of points added is
                            4 + 4*(boundary_points_per_edge).

    Returns:
      merged_control_point_locations: augmented set of control point locations
      merged_control_point_flows: augmented set of control point flows
    """

    batch_size = tf.compat.dimension_value(control_point_locations.shape[0])

    boundary_point_locations = _get_boundary_locations(
        image_height, image_width, boundary_points_per_edge)

    boundary_point_flows = tf.zeros([tf.shape(boundary_point_locations)[0], 2])

    type_to_use = control_point_locations.dtype
    boundary_point_locations = tf.cast(
        _expand_to_minibatch(boundary_point_locations, batch_size), type_to_use)

    boundary_point_flows = tf.cast(
        _expand_to_minibatch(boundary_point_flows, batch_size), type_to_use)

    merged_control_point_locations = tf.concat(
        [control_point_locations, boundary_point_locations], 1)

    merged_control_point_flows = tf.concat(
        [control_point_flows, boundary_point_flows], 1)

    return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp(image,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundary_points=0,
                      name='sparse_image_warp'):
    """Image warping using correspondences between sparse control points.

    Apply a non-linear warp to the image, where the warp is specified by
    the source and destination locations of a (potentially small) number of
    control points. First, we use a polyharmonic spline
    (`tf.contrib.image.interpolate_spline`) to interpolate the displacements
    between the corresponding control points to a dense flow field.
    Then, we warp the image using this dense flow field
    (`tf.contrib.image.dense_image_warp`).

    Let t index our control points. For regularization_weight=0, we have:
    warped_image[b, dest_control_point_locations[b, t, 0],
                    dest_control_point_locations[b, t, 1], :] =
    image[b, source_control_point_locations[b, t, 0],
             source_control_point_locations[b, t, 1], :].

    For regularization_weight > 0, this condition is met approximately, since
    regularized interpolation trades off smoothness of the interpolant vs.
    reconstruction of the interpolant at the control points.
    See `tf.contrib.image.interpolate_spline` for further documentation of the
    interpolation_order and regularization_weight arguments.


    Args:
      image: `[batch, height, width, channels]` float `Tensor`
      source_control_point_locations: `[batch, num_control_points, 2]` float
        `Tensor`
      dest_control_point_locations: `[batch, num_control_points, 2]` float
        `Tensor`
      interpolation_order: polynomial order used by the spline interpolation
      regularization_weight: weight on smoothness regularizer in interpolation
      num_boundary_points: How many zero-flow boundary points to include at
        each image edge.Usage:
          num_boundary_points=0: don't add zero-flow points
          num_boundary_points=1: 4 corners of the image
          num_boundary_points=2: 4 corners and one in the middle of each edge
            (8 points total)
          num_boundary_points=n: 4 corners and n-1 along each edge
      name: A name for the operation (optional).

      Note that image and offsets can be of type tf.half, tf.float32, or
      tf.float64, and do not necessarily have to be the same type.

    Returns:
      warped_image: `[batch, height, width, channels]` float `Tensor` with same
        type as input image.
      flow_field: `[batch, height, width, 2]` float `Tensor` containing the
        dense flow field produced by the interpolation.
    """

    image = tf.convert_to_tensor(image)
    source_control_point_locations = tf.convert_to_tensor(
        source_control_point_locations)
    dest_control_point_locations = tf.convert_to_tensor(
        dest_control_point_locations)

    control_point_flows = (
        dest_control_point_locations - source_control_point_locations)

    clamp_boundaries = num_boundary_points > 0
    boundary_points_per_edge = num_boundary_points - 1

    with tf.name_scope(name or "sparse_image_warp"):

        image_shape = tf.shape(image)
        batch_size, image_height, image_width = (image_shape[0], image_shape[1],
                                                 image_shape[2])

        # This generates the dense locations where the interpolant
        # will be evaluated.
        grid_locations = _get_grid_locations(image_height, image_width)

        flattened_grid_locations = tf.reshape(grid_locations,
                                              [image_height * image_width, 2])

        flattened_grid_locations = tf.cast(
            _expand_to_minibatch(flattened_grid_locations, batch_size),
            image.dtype)

        if clamp_boundaries:
            (dest_control_point_locations,
             control_point_flows) = _add_zero_flow_controls_at_boundary(
                 dest_control_point_locations, control_point_flows,
                 image_height, image_width, boundary_points_per_edge)

        flattened_flows = interpolate_spline.interpolate_spline(
            dest_control_point_locations, control_point_flows,
            flattened_grid_locations, interpolation_order,
            regularization_weight)

        dense_flows = tf.reshape(flattened_flows,
                                 [batch_size, image_height, image_width, 2])

        warped_image = dense_image_warp(image, dense_flows)

        return warped_image, dense_flows
