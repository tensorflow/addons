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

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.embedding_bag import EmbeddingBag, _embedding_bag
from tensorflow_addons.utils import test_utils


def manual_embedding_bag(indices, params, weights=None, combiner="mean"):
    gathered = tf.gather(params, indices)
    if weights is not None:
        gathered *= tf.expand_dims(weights, -1)
    if combiner == "sum":
        return tf.reduce_sum(gathered, -2, keepdims=False)
    else:
        assert combiner == 'mean'
        assert weights is None
        return tf.reduce_mean(gathered, -2, keepdims=False)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("input_shape", [(16, 32)])
@pytest.mark.parametrize("input_dim", [63, 64])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("indices_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("combiner", ["sum", "mean"])
def test_forward(input_shape, input_dim, dtype, indices_dtype, combiner):
    indices = np.random.randint(low=0, high=input_dim, size=input_shape).astype(
        indices_dtype
    )
    params = np.random.random(size=(input_dim, 16)).astype(dtype)
    if combiner == 'sum':
        weights = np.random.random(size=indices.shape).astype(dtype)
    else:
        weights = None
    expected = manual_embedding_bag(indices, params, weights, combiner=combiner)
    embedding_bag = EmbeddingBag(input_dim, 16, combiner=combiner, dtype=dtype)
    embedding_bag.build(indices.shape)
    embedding_bag.set_weights([params])
    indices = tf.convert_to_tensor(indices)
    if weights is not None:
        weights = tf.convert_to_tensor(weights)
    output = embedding_bag(
        indices,
        weights,
    )
    test_utils.assert_allclose_according_to_type(expected, output)


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("input_shape", [(16, 32)])
@pytest.mark.parametrize("input_dim", [63, 64])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("indices_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("combiner", ["sum", "mean"])
def test_backward(input_shape, input_dim, dtype, indices_dtype, combiner):
    indices = np.random.randint(low=0, high=input_dim, size=input_shape).astype(
        indices_dtype
    )
    params = np.random.random(size=(input_dim, 16)).astype(dtype)
    if combiner == 'sum':
        weights = np.random.random(size=indices.shape).astype(dtype)
    else:
        weights = None

    indices = tf.convert_to_tensor(indices)
    params = tf.convert_to_tensor(params)
    if weights is not None:
        weights = tf.convert_to_tensor(weights)

    if combiner == 'sum':
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([params, weights])
            output = _embedding_bag(indices, params, weights, combiner='sum')
            expected = manual_embedding_bag(indices, params, weights, combiner='sum')

        grads = tape.gradient(output, [params, weights])
        expected_grads = tape.gradient(expected, [params, weights])
        # Gather returns sparse IndexedSlices so we have to sum them together.
        test_utils.assert_allclose_according_to_type(
            tf.math.unsorted_segment_sum(
                expected_grads[0].values,
                expected_grads[0].indices,
                expected_grads[0].dense_shape[0],
            ),
            tf.math.unsorted_segment_sum(
                grads[0].values,
                grads[0].indices,
                grads[0].dense_shape[0],
            ),
        )
        test_utils.assert_allclose_according_to_type(
            expected_grads[1],
            grads[1],
        )
    else:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(params)
            output = _embedding_bag(indices, params, combiner=combiner)
            expected = manual_embedding_bag(indices, params, combiner=combiner)

        grads = tape.gradient(output, [params])
        expected_grads = tape.gradient(expected, [params])
        # Gather returns sparse IndexedSlices so we have to sum them together.
        test_utils.assert_allclose_according_to_type(
            tf.math.unsorted_segment_sum(
                expected_grads[0].values,
                expected_grads[0].indices,
                expected_grads[0].dense_shape[0],
            ),
            tf.math.unsorted_segment_sum(
                grads[0].values,
                grads[0].indices,
                grads[0].dense_shape[0],
            ),
        )
