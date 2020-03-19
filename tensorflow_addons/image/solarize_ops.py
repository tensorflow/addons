""" This module is used to invert all pixel values above a threshold
   which simply means segmentation. """

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


def solarize(image: TensorLike, threshold: float = 128) -> TensorLike:

    """Method to solarize the image
  image: input image
  threshold: threshold value to solarize the image

  Returns:
    A solarized image
  """
    # For each pixel in the image, select the pixel
    # if the value is less than the threshold.
    # Otherwise, subtract 255 from the pixel.
    return tf.where(image < threshold, image, 255 - image)


def solarize_add(
    image: TensorLike, addition: int = 0, threshold: float = 128
) -> TensorLike:
    """Method to add solarize to the image
  image: input image
  addition: addition amount to add in image
  threshold: threshold value to solarize the image

  Returns:
    Solarized image with addition values
  """
    # For each pixel in the image less than threshold
    # we add 'addition' amount to it and then clip the
    # pixel value to be between 0 and 255. The value
    # of 'addition' is between -128 and 128.
    added_image = tf.cast(image, tf.int64) + addition
    added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
    return tf.where(image < threshold, added_image, image)
