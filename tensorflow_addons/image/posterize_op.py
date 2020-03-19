"""Posterise the image.
  Means conversion of 
  continuous gradation of tone to several
  regions of fewer tones"""
  
import tensorflow as tf

def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
