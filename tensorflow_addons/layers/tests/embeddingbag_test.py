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


import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.embeddingbag import embeddingbag
from tensorflow_addons.utils import test_utils


def manual_embeddingbag(indices, values, weights):
    gathered = tf.gather(values, indices)  # (batch_dims, bag_size, value_dim)
    gathered *= tf.expand_dims(weights, -1)  # (batch_dims, bag_size, value_dim)
    return tf.reduce_sum(gathered, -2, keepdims=False)  # (batch_dims, key_dim)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_simple(dtype):
    indices = np.random.randint(low=0, high=1024, size=(16, 32)).astype(dtype)
    values = np.random.random(size=(1024, 16)).astype(np.float32)
    weights = np.random.random(size=indices.shape).astype(np.float32)
    manual_output = manual_embeddingbag(indices, values, weights)
    fused_output = embeddingbag(indices, values, weights)
    test_utils.assert_allclose_according_to_type(
        manual_output, fused_output
    )
