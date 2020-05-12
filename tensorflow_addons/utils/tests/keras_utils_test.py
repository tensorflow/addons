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
"""Tests for Keras utils."""

import sys

import pytest
import tensorflow as tf

from tensorflow_addons.utils import keras_utils


def test_normalize_data_format():
    assert keras_utils.normalize_data_format("Channels_Last") == "channels_last"
    assert keras_utils.normalize_data_format("CHANNELS_FIRST") == "channels_first"

    with pytest.raises(ValueError, match="The `data_format` argument must be one of"):
        keras_utils.normalize_data_format("invalid")


def test_normalize_tuple():
    assert (2, 2, 2) == keras_utils.normalize_tuple(2, n=3, name="strides")
    assert (2, 1, 2) == keras_utils.normalize_tuple((2, 1, 2), n=3, name="strides")

    with pytest.raises(ValueError):
        keras_utils.normalize_tuple((2, 1), n=3, name="strides")

    with pytest.raises(TypeError):
        keras_utils.normalize_tuple(None, n=3, name="strides")


def test_standard_cell():
    keras_utils.assert_like_rnncell("cell", tf.keras.layers.LSTMCell(10))


def test_non_cell():
    with pytest.raises(TypeError):
        keras_utils.assert_like_rnncell("cell", tf.keras.layers.Dense(10))


def test_custom_cell():
    class CustomCell(tf.keras.layers.AbstractRNNCell):
        @property
        def output_size(self):
            raise ValueError("assert_like_rnncell should not run code")

    keras_utils.assert_like_rnncell("cell", CustomCell())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
