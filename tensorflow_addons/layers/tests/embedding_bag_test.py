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
"""Tests for EmbeddingBag layer."""

# TODO Add a test for the layer separate from the op
# TODO Test gradients as well
# TODO Test a few shapes with weird dimensions to make sure the op handles them correctly

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.embedding_bag import EmbeddingBag
from tensorflow_addons.utils import test_utils


def manual_embedding_bag(indices, values, weights):
    gathered = tf.gather(values, indices)  # (batch_dims, bag_size, value_dim)
    gathered *= tf.expand_dims(weights, -1)  # (batch_dims, bag_size, value_dim)
    return tf.reduce_sum(gathered, -2, keepdims=False)  # (batch_dims, key_dim)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("indices_dtype", [np.int32, np.int64])
def test_simple(dtype, indices_dtype):
    indices = np.random.randint(low=0, high=1024, size=(16, 32)).astype(indices_dtype)
    values = np.random.random(size=(1024, 16)).astype(dtype)
    weights = np.random.random(size=indices.shape).astype(dtype)
    manual_output = manual_embedding_bag(indices, values, weights)
    fused_embedding_bag = EmbeddingBag(1024, 16, dtype=dtype)
    fused_embedding_bag.build(indices.shape)
    fused_embedding_bag.set_weights([values])
    fused_output = fused_embedding_bag(indices, weights)
    test_utils.assert_allclose_according_to_type(manual_output, fused_output)
