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
"""Utilities for tf.keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def register_keras_custom_object(cls):
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.

    Ported from tensorflow.python.keras.util.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable
        of ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name +
                             '` argument must be a tuple of ' + str(n) +
                             ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name +
                             '` argument must be a tuple of ' + str(n) +
                             ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError('The `' + name +
                                 '` argument must be a tuple of ' + str(n) +
                                 ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) +
                                 ' of type' + ' ' + str(type(single_value)))
        return value_tuple
