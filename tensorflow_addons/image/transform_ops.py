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
"""Image transform ops."""

import tensorflow as tf
from tensorflow_addons.image import utils as img_utils
from tensorflow_addons.utils.types import TensorLike
from tensorflow_addons.image.utils import wrap, unwrap

from typing import Optional


_IMAGE_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


def transform(
    images: TensorLike,
    transforms: TensorLike,
    interpolation: str = "NEAREST",
    fill_mode: str = "CONSTANT",
    output_shape: Optional[list] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Applies the given transform(s) to the image(s).

    Args:
      images: A tensor of shape (num_images, num_rows, num_columns,
        num_channels) (NHWC), (num_rows, num_columns, num_channels) (HWC), or
        (num_rows, num_columns) (HW).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is
        [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
        `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to
        the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      interpolation: Interpolation mode.
        Supported values: "NEAREST", "BILINEAR".
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
        - *reflect*: `(d c b a | a b c d | d c b a)`
          The input is extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)`
          The input is extended by filling all values beyond the edge with the
          same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)`
          The input is extended by wrapping around to the opposite edge.
      output_shape: Output dimesion after the transform, [height, width].
        If None, output is the same size as input image.

      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.

    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with tf.name_scope(name or "transform"):
        image_or_images = tf.convert_to_tensor(images, name="images")
        transform_or_transforms = tf.convert_to_tensor(
            transforms, name="transforms", dtype=tf.dtypes.float32
        )
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = img_utils.to_4D_image(image_or_images)
        original_ndims = img_utils.get_ndims(image_or_images)

        if output_shape is None:
            output_shape = tf.shape(images)[1:3]

        output_shape = tf.convert_to_tensor(
            output_shape, tf.dtypes.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width"
            )

        if len(transform_or_transforms.get_shape()) == 1:
            transforms = transform_or_transforms[None]
        elif transform_or_transforms.get_shape().ndims is None:
            raise ValueError("transforms rank must be statically known")
        elif len(transform_or_transforms.get_shape()) == 2:
            transforms = transform_or_transforms
        else:
            transforms = transform_or_transforms
            raise ValueError(
                "transforms should have rank 1 or 2, but got rank %d"
                % len(transforms.get_shape())
            )

        # TODO(WindQAQ): Support "nearest" `fill_mode` and `fill_value` in TF2.4.
        output = tf.raw_ops.ImageProjectiveTransformV2(
            images=images,
            transforms=transforms,
            output_shape=output_shape,
            interpolation=interpolation.upper(),
            fill_mode=fill_mode.upper(),
        )
        return img_utils.from_4D_image(output, original_ndims)


def compose_transforms(transforms: TensorLike, name: Optional[str] = None) -> tf.Tensor:
    """Composes the transforms tensors.

    Args:
      transforms: List of image projective transforms to be composed. Each
        transform is length 8 (single transform) or shape (N, 8) (batched
        transforms). The shapes of all inputs must be equal, and at least one
        input must be given.
      name: The name for the op.

    Returns:
      A composed transform tensor. When passed to `transform` op,
          equivalent to applying each of the given transforms to the image in
          order.
    """
    assert transforms, "transforms cannot be empty"
    with tf.name_scope(name or "compose_transforms"):
        composed = flat_transforms_to_matrices(transforms[0])
        for tr in transforms[1:]:
            # Multiply batches of matrices.
            composed = tf.matmul(composed, flat_transforms_to_matrices(tr))
        return matrices_to_flat_transforms(composed)


def flat_transforms_to_matrices(
    transforms: TensorLike, name: Optional[str] = None
) -> tf.Tensor:
    """Converts projective transforms to affine matrices.

    Note that the output matrices map output coordinates to input coordinates.
    For the forward transformation matrix, call `tf.linalg.inv` on the result.

    Args:
      transforms: Vector of length 8, or batches of transforms with shape
        `(N, 8)`.
      name: The name for the op.

    Returns:
      3D tensor of matrices with shape `(N, 3, 3)`. The output matrices map the
        *output coordinates* (in homogeneous coordinates) of each transform to
        the corresponding *input coordinates*.

    Raises:
      ValueError: If `transforms` have an invalid shape.
    """
    with tf.name_scope(name or "flat_transforms_to_matrices"):
        transforms = tf.convert_to_tensor(transforms, name="transforms")
        if transforms.shape.ndims not in (1, 2):
            raise ValueError("Transforms should be 1D or 2D, got: %s" % transforms)
        # Make the transform(s) 2D in case the input is a single transform.
        transforms = tf.reshape(transforms, tf.constant([-1, 8]))
        num_transforms = tf.shape(transforms)[0]
        # Add a column of ones for the implicit last entry in the matrix.
        return tf.reshape(
            tf.concat([transforms, tf.ones([num_transforms, 1])], axis=1),
            tf.constant([-1, 3, 3]),
        )


def matrices_to_flat_transforms(
    transform_matrices: TensorLike, name: Optional[str] = None
) -> tf.Tensor:
    """Converts affine matrices to projective transforms.

    Note that we expect matrices that map output coordinates to input
    coordinates. To convert forward transformation matrices,
    call `tf.linalg.inv` on the matrices and use the result here.

    Args:
      transform_matrices: One or more affine transformation matrices, for the
        reverse transformation in homogeneous coordinates. Shape `(3, 3)` or
        `(N, 3, 3)`.
      name: The name for the op.

    Returns:
      2D tensor of flat transforms with shape `(N, 8)`, which may be passed
      into `transform` op.

    Raises:
      ValueError: If `transform_matrices` have an invalid shape.
    """
    with tf.name_scope(name or "matrices_to_flat_transforms"):
        transform_matrices = tf.convert_to_tensor(
            transform_matrices, name="transform_matrices"
        )
        if transform_matrices.shape.ndims not in (2, 3):
            raise ValueError(
                "Matrices should be 2D or 3D, got: %s" % transform_matrices
            )
        # Flatten each matrix.
        transforms = tf.reshape(transform_matrices, tf.constant([-1, 9]))
        # Divide each matrix by the last entry (normally 1).
        transforms /= transforms[:, 8:9]
        return transforms[:, :8]


def angles_to_projective_transforms(
    angles: TensorLike,
    image_height: TensorLike,
    image_width: TensorLike,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Returns projective transform(s) for the given angle(s).

    Args:
      angles: A scalar angle to rotate all images by, or (for batches of
        images) a vector with an angle to rotate each image in the batch. The
        rank must be statically known (the shape is not `TensorShape(None)`.
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.

    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
      given to `transform` op.
    """
    with tf.name_scope(name or "angles_to_projective_transforms"):
        angle_or_angles = tf.convert_to_tensor(
            angles, name="angles", dtype=tf.dtypes.float32
        )
        if len(angle_or_angles.get_shape()) == 0:
            angles = angle_or_angles[None]
        elif len(angle_or_angles.get_shape()) == 1:
            angles = angle_or_angles
        else:
            raise ValueError("angles should have rank 0 or 1.")
        cos_angles = tf.math.cos(angles)
        sin_angles = tf.math.sin(angles)
        x_offset = (
            (image_width - 1)
            - (cos_angles * (image_width - 1) - sin_angles * (image_height - 1))
        ) / 2.0
        y_offset = (
            (image_height - 1)
            - (sin_angles * (image_width - 1) + cos_angles * (image_height - 1))
        ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                cos_angles[:, None],
                -sin_angles[:, None],
                x_offset[:, None],
                sin_angles[:, None],
                cos_angles[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.dtypes.float32),
            ],
            axis=1,
        )


def rotate(
    images: TensorLike,
    angles: TensorLike,
    interpolation: str = "NEAREST",
    fill_mode: str = "CONSTANT",
    name: Optional[str] = None,
) -> tf.Tensor:
    """Rotate image(s) counterclockwise by the passed angle(s) in radians.

    Args:
      images: A tensor of shape
        `(num_images, num_rows, num_columns, num_channels)`
        (NHWC), `(num_rows, num_columns, num_channels)` (HWC), or
        `(num_rows, num_columns)` (HW).
      angles: A scalar angle to rotate all images by, or (if `images` has rank 4)
        a vector of length num_images, with an angle for each image in the
        batch.
      interpolation: Interpolation mode. Supported values: "NEAREST",
        "BILINEAR".
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{'constant', 'reflect', 'wrap'}`).
        - *reflect*: `(d c b a | a b c d | d c b a)`
          The input is extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)`
          The input is extended by filling all values beyond the edge with the
          same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)`
          The input is extended by wrapping around to the opposite edge.
      name: The name of the op.

    Returns:
      Image(s) with the same type and shape as `images`, rotated by the given
      angle(s). Empty space due to the rotation will be filled with zeros.

    Raises:
      TypeError: If `images` is an invalid type.
    """
    with tf.name_scope(name or "rotate"):
        image_or_images = tf.convert_to_tensor(images)
        if image_or_images.dtype.base_dtype not in _IMAGE_DTYPES:
            raise TypeError("Invalid dtype %s." % image_or_images.dtype)
        images = img_utils.to_4D_image(image_or_images)
        original_ndims = img_utils.get_ndims(image_or_images)

        image_height = tf.cast(tf.shape(images)[1], tf.dtypes.float32)[None]
        image_width = tf.cast(tf.shape(images)[2], tf.dtypes.float32)[None]
        output = transform(
            images,
            angles_to_projective_transforms(angles, image_height, image_width),
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        return img_utils.from_4D_image(output, original_ndims)


def shear_x(image: TensorLike, level: float, replace: TensorLike) -> TensorLike:
    """Perform shear operation on an image (x-axis).

    Args:
        image: A 3D image `Tensor`.
        level: A float denoting shear element along y-axis
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        Transformed image along X or Y axis, with space outside image
        filled with replace.
    """
    # Shear parallel to x axis is a projective transform
    # with a matrix form of:
    # [1  level
    #  0  1].
    image = transform(wrap(image), [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return unwrap(image, replace)


def shear_y(image: TensorLike, level: float, replace: TensorLike) -> TensorLike:
    """Perform shear operation on an image (y-axis).

    Args:
        image: A 3D image `Tensor`.
        level: A float denoting shear element along x-axis
        replace: A one or three value 1D tensor to fill empty pixels.
    Returns:
        Transformed image along X or Y axis, with space outside image
        filled with replace.
    """
    # Shear parallel to y axis is a projective transform
    # with a matrix form of:
    # [1  0
    #  level  1].
    image = transform(wrap(image), [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0])
    return unwrap(image, replace)
