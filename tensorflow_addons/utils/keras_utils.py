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

import tensorflow as tf


def normalize_data_format(value):
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            '"channels_first", "channels_last". Received: ' + str(value)
        )
    return data_format


def normalize_tuple(value, n, name):
    """Transforms an integer or iterable of integers into an integer tuple.

    A copy of tensorflow.python.keras.util.

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
            raise TypeError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        if len(value_tuple) != n:
            raise ValueError(
                "The `"
                + name
                + "` argument must be a tuple of "
                + str(n)
                + " integers. Received: "
                + str(value)
            )
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError(
                    "The `"
                    + name
                    + "` argument must be a tuple of "
                    + str(n)
                    + " integers. Received: "
                    + str(value)
                    + " "
                    "including element "
                    + str(single_value)
                    + " of type"
                    + " "
                    + str(type(single_value))
                )
        return value_tuple


def _hasattr(obj, attr_name):
    # If possible, avoid retrieving the attribute as the object might run some
    # lazy computation in it.
    if attr_name in dir(obj):
        return True
    try:
        getattr(obj, attr_name)
    except AttributeError:
        return False
    else:
        return True


def assert_like_rnncell(cell_name, cell):
    """Raises a TypeError if cell is not like a
    tf.keras.layers.AbstractRNNCell.

    Args:
      cell_name: A string to give a meaningful error referencing to the name
        of the function argument.
      cell: The object which should behave like a
        tf.keras.layers.AbstractRNNCell.

    Raises:
      TypeError: A human-friendly exception.
    """
    conditions = [
        _hasattr(cell, "output_size"),
        _hasattr(cell, "state_size"),
        _hasattr(cell, "get_initial_state"),
        callable(cell),
    ]

    errors = [
        "'output_size' property is missing",
        "'state_size' property is missing",
        "'get_initial_state' method is required",
        "is not callable",
    ]

    if not all(conditions):
        errors = [error for error, cond in zip(errors, conditions) if not cond]
        raise TypeError(
            "The argument {!r} ({}) is not an RNNCell: {}.".format(
                cell_name, cell, ", ".join(errors)
            )
        )
