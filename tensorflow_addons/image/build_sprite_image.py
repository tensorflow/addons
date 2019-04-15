from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


@tf.function
def build_sprite_image(images, size=None):
    """Builds a sprite image out of all the images passed in the arg. It will
    output an image with the same height and width.
    If there are lesser number of images it adds blank images in the end.
    :param images: 4-D Tensor of shape `[batch, height, width, channels]`

    :param size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.
          The new size for the images.

    :raises ValueError: if the shape of `images` is incompatible with the
              shape arguments to this function
            ValueError: size is not a list of length 2.
            ValueError: size is not a list consisting of integers.

    :return: A Tensor '[new_dim, new_dim, channels]'."""

    dimension = int(np.ceil(np.sqrt(np.shape(images)[0])))
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    images_shape = images.get_shape()

    if images_shape.ndims != 4:
        raise ValueError('\'image\' must have 4 dimensions.')
    if size is not None:
        if (np.size(size) != 2 and np.size(size) != 0):
            raise ValueError('\'size\' must be a list of size 2, [height,width].')

        if np.size(size) == 2:
            if (isinstance(size[0], int) is False or isinstance(size[1], int) is False):
                raise TypeError('\'size\' must be a list of two integers.')
        images = tf.image.resize_images(images, [size[0], size[1]])
        images_shape = images.get_shape()

    images_shape = tf.cast(images_shape, dtype=tf.float32)
    dim = tf.math.ceil(tf.sqrt(images_shape[0]))
    images_shape = tf.cast(images_shape, dtype=tf.int32)

    if dim * dim - tf.cast(images_shape[0], dtype=tf.float32) is not 0:
        first = tf.cast(dim * dim - tf.cast(images_shape[0], dtype=tf.float32), tf.int32)
        shp = [first, images_shape[1], images_shape[2], images_shape[3]]
        temp = tf.zeros(shape=shp)
        images = tf.concat([images, temp], axis=0)

    final = []

    for iterator in range(dimension):
        temp = [images[(iterator * dimension) + i] for i in range(dimension)]
        row = tf.concat(temp, axis=1)
        if iterator == 0:
            final = row
        else:
            final = tf.concat([final, row], axis=0)

    return tf.convert_to_tensor(final, dtype=tf.float32)
