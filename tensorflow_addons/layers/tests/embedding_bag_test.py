# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for EmbeddingBag layer."""

# TODO Add a test for the layer separate from the op
# TODO Test gradients as well
# TODO Test a few shapes with weird dimensions to make sure the op handles them correctly

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.embedding_bag import EmbeddingBag, _embedding_bag
from tensorflow_addons.utils import test_utils


def manual_embedding_bag(indices, values, weights):
    gathered = tf.gather(values, indices)  # (batch_dims, bag_size, value_dim)
    gathered *= tf.expand_dims(weights, -1)  # (batch_dims, bag_size, value_dim)
    return tf.reduce_sum(gathered, -2, keepdims=False)  # (batch_dims, key_dim)


@pytest.mark.parametrize("input_shape", [(16, 32), (16, 8, 32)])
@pytest.mark.parametrize("input_dim", [3, 512, 1024])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("indices_dtype", [np.int32, np.int64])
def test_foward(input_shape, input_dim, dtype, indices_dtype):
    indices = np.random.randint(low=0, high=input_dim, size=input_shape).astype(
        indices_dtype
    )
    values = np.random.random(size=(input_dim, 16)).astype(dtype)
    weights = np.random.random(size=indices.shape).astype(dtype)
    expected = manual_embedding_bag(indices, values, weights)
    embedding_bag = EmbeddingBag(input_dim, 16, dtype=dtype)
    embedding_bag.build(indices.shape)
    embedding_bag.set_weights([values])
    output = embedding_bag(indices, weights)
    test_utils.assert_allclose_according_to_type(expected, output)


@pytest.mark.parametrize("input_shape", [(16, 32), (16, 8, 32)])
@pytest.mark.parametrize("input_dim", [3, 512, 1024])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("indices_dtype", [np.int32, np.int64])
def test_backward(input_shape, input_dim, dtype, indices_dtype):
    indices = np.random.randint(low=0, high=input_dim, size=input_shape).astype(
        indices_dtype
    )
    values = np.random.random(size=(input_dim, 16)).astype(dtype)
    weights = np.random.random(size=indices.shape).astype(dtype)

    indices = tf.convert_to_tensor(indices)
    values = tf.convert_to_tensor(values)
    weights = tf.convert_to_tensor(weights)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([values, weights])
        output = _embedding_bag(indices, values, weights)
        expected = manual_embedding_bag(indices, values, weights)

    grads = tape.gradient(output, [values, weights])
    expected_grads = tape.gradient(expected, [values, weights])
    # Gather returns sparse IndexedSlices so we have to sum them together.
    test_utils.assert_allclose_according_to_type(
        tf.math.unsorted_segment_sum(
            expected_grads[0].values,
            expected_grads[0].indices,
            expected_grads[0].dense_shape[0],
        ),
        grads[0],
    )
    test_utils.assert_allclose_according_to_type(
        expected_grads[1],
        grads[1],
    )
