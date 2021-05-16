# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for MaxUnpooling2D layers."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow_addons.layers.max_unpooling_2d import MaxUnpooling2D


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_simple():
    valid_input = np.array([13, 4]).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 1, 2, 1))
    indices = np.array([1, 6]).astype(np.float32)
    indices = np.reshape(indices, (1, 1, 2, 1))
    expected_output_shape = (1, 2, 4, 1)
    expected_output = np.array([0, 13, 0, 0, 0, 0, 4, 0]).astype(np.float32)
    expected_output = np.reshape(expected_output, expected_output_shape)

    output = MaxUnpooling2D()(valid_input, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_strides2x1():
    valid_input = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 2, 2, 2))
    indices = np.array([0, 3, 4, 7, 8, 11, 12, 15]).astype(np.float32)
    indices = np.reshape(indices, (1, 2, 2, 2))
    expected_output_shape = (1, 4, 2, 2)
    expected_output = np.array([1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6, 7, 0, 0, 8]).astype(
        np.float32
    )
    expected_output = np.reshape(expected_output, expected_output_shape)

    output = MaxUnpooling2D(strides=(2, 1))(valid_input, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_strides2x2():
    valid_input = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 2, 4, 1))
    indices = np.array([0, 5, 10, 13, 19, 20, 27, 31]).astype(np.float32)
    indices = np.reshape(indices, (1, 2, 4, 1))
    expected_output_shape = (1, 4, 8, 1)
    expected_output = np.array(
        # fmt: off
        [
            1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0,
            0, 0, 7, 0, 0, 0, 8
        ]
        # fmt: on
    ).astype(np.float32)
    expected_output = np.reshape(expected_output, expected_output_shape)

    output = MaxUnpooling2D()(valid_input, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_batch():
    valid_input = np.array(
        # fmt: off
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
        # fmt: on
    ).astype(np.float32)
    valid_input = np.reshape(valid_input, (2, 2, 4, 2))
    indices = np.array(
        # fmt: off
        [
            2, 23, 8, 9, 12, 15, 40, 43, 44, 47, 72, 75, 80, 79, 62, 65, 0, 1, 30, 7,
            14, 35, 42, 21, 68, 69, 50, 51, 56, 5, 86, 63
        ]
        # fmt: on
    ).astype(np.float32)
    indices = np.reshape(indices, (2, 2, 4, 2))
    expected_output_shape = (2, 4, 12, 2)
    expected_output = np.array(
        # fmt: off
        [
            0, 0, 1, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 2, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 8, 9, 0, 0, 10, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 16, 0, 0, 0, 0, 0, 0, 11,
            0, 0, 12, 0, 0, 0, 14, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            17, 18, 0, 0, 0, 30, 0, 20, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 24, 0,
            0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0,
            0, 0, 0, 27, 28, 0, 0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 25, 26,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0,
            0, 0
        ]
        # fmt: on
    ).astype(np.float32)
    expected_output = np.reshape(expected_output, expected_output_shape)

    output = MaxUnpooling2D(strides=(2, 3))(valid_input, indices)
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_batch_and_padding_valid():
    valid_input = np.array(
        # fmt: off
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
        # fmt: on
    ).astype(np.float32)
    valid_input = np.reshape(valid_input, (2, 2, 4, 2))
    indices = np.array(
        # fmt: off
        [
            2, 23, 8, 9, 12, 15, 40, 43, 44, 47, 72, 75, 80, 79, 62, 65, 0, 1, 30, 7,
            14, 35, 42, 21, 68, 69, 50, 51, 56, 5, 86, 63
        ]
        # fmt: on
    ).astype(np.float32)
    indices = np.reshape(indices, (2, 2, 4, 2))
    expected_output_shape = (2, 4, 11, 2)
    expected_output = np.array(
        # fmt: off
        [
            0, 0, 1, 0, 0, 0, 0, 0, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 2, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 8, 9, 0, 0, 10, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 16, 0, 0, 0, 0, 0, 0, 11,
            0, 0, 12, 0, 0, 0, 14, 13, 0, 0, 0, 0, 0, 0, 0, 17, 18, 0, 0, 0, 30, 0,
            20, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0,
            19, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 27, 28, 0,
            0, 0, 0, 29, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 25, 26, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0
        ]
        # fmt: on
    ).astype(np.float32)
    expected_output = np.reshape(expected_output, expected_output_shape)

    output = MaxUnpooling2D(strides=(2, 3), padding="VALID")(valid_input, indices)
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_with_pooling_simple():
    valid_input = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 2, 4, 1))
    updates, indices = tf.nn.max_pool_with_argmax(
        valid_input, ksize=[2, 2], strides=[2, 2], padding="SAME"
    )
    expected_output = np.array([0, 0, 0, 0, 0, 6, 0, 8]).astype(np.float32)
    expected_output = np.reshape(expected_output, valid_input.shape)

    output = MaxUnpooling2D()(updates, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_with_pooling():
    valid_input = np.array(
        [1, 2, 4, 3, 8, 6, 7, 5, 9, 10, 12, 11, 13, 16, 15, 14]
    ).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 4, 1))
    updates, indices = tf.nn.max_pool_with_argmax(
        valid_input, ksize=[2, 2], strides=[2, 2], padding="SAME"
    )
    expected_output = np.array(
        [0, 0, 0, 0, 8, 0, 7, 0, 0, 0, 0, 0, 0, 16, 15, 0]
    ).astype(np.float32)
    expected_output = np.reshape(expected_output, valid_input.shape)

    output = MaxUnpooling2D()(updates, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_strides_different_from_filter():
    valid_input = np.array(
        [1, 2, 4, 3, 8, 6, 7, 5, 9, 10, 12, 11, 13, 16, 15, 14]
    ).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 4, 1))
    updates, indices = tf.nn.max_pool_with_argmax(
        valid_input, ksize=[1, 2, 2, 1], strides=[1, 4, 4, 1], padding="SAME"
    )
    expected_output = np.array([0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(
        np.float32
    )
    expected_output = np.reshape(expected_output, valid_input.shape)

    output = MaxUnpooling2D(strides=(4, 4))(updates, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_padding_valid():
    valid_input = np.array(
        [1, 2, 4, 3, 8, 6, 7, 5, 9, 10, 12, 11, 13, 16, 15, 14]
    ).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 4, 1))
    updates, indices = tf.nn.max_pool_with_argmax(
        valid_input, ksize=[2, 2], strides=[2, 2], padding="VALID"
    )
    expected_output = np.array(
        [0, 0, 0, 0, 8, 0, 7, 0, 0, 0, 0, 0, 0, 16, 15, 0]
    ).astype(np.float32)
    expected_output = np.reshape(expected_output, valid_input.shape)

    output = MaxUnpooling2D(padding="VALID")(updates, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_padding_valid_complex():
    valid_input = np.array(
        [1, 2, 4, 3, 8, 6, 7, 5, 9, 10, 12, 11, 13, 16, 15, 14, 20, 18, 17, 19]
    ).astype(np.float32)
    valid_input = np.reshape(valid_input, (1, 4, 5, 1))
    updates, indices = tf.nn.max_pool_with_argmax(
        valid_input, ksize=[2, 3], strides=[2, 2], padding="VALID"
    )
    expected_output = np.array(
        [0, 0, 0, 0, 0, 0, 7, 0, 0, 10, 0, 0, 0, 0, 0, 0, 20, 0, 0, 19]
    ).astype(np.float32)
    expected_output = np.reshape(expected_output, valid_input.shape)

    output = MaxUnpooling2D(pool_size=(2, 3), padding="VALID")(updates, indices).numpy()
    np.testing.assert_array_equal(expected_output, output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_symbolic_tensor_shape():
    valid_input = tf.keras.layers.Input((None, None, 1))
    updates, indices = tf.nn.max_pool_with_argmax(
        valid_input, ksize=[2, 2], strides=[2, 2], padding="SAME"
    )
    output = MaxUnpooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(
        updates, indices
    )
    np.testing.assert_array_equal(valid_input.shape.as_list(), output.shape.as_list())
