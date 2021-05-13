# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import pytest

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import rrelu
from tensorflow_addons.utils import test_utils

SEED = 111111


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu_old(dtype, training):
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
    lower = 0.1
    upper = 0.2

    tf.random.set_seed(SEED)
    training_results = {
        np.float16: [-0.288330078, -0.124206543, 0, 1, 2],
        np.float32: [-0.26851666, -0.116421416, 0, 1, 2],
        np.float64: [-0.3481333923206531, -0.17150176242558851, 0, 1, 2],
    }
    result = rrelu(x, lower, upper, training=training, seed=SEED)
    if training:
        expect_result = training_results.get(dtype)
    else:
        expect_result = [
            -0.30000001192092896,
            -0.15000000596046448,
            0,
            1,
            2,
        ]
    test_utils.assert_allclose_according_to_type(result, expect_result)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("training", [True, False])
def test_rrelu(dtype, training):
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
    lower = 0.1
    upper = 0.2
    training_results = {
        np.float16: [-0.3826, -0.165, 0, 1, 2],
        np.float32: [-0.282151192, -0.199812651, 0, 1, 2],
        np.float64: [-0.25720977, -0.1221586, 0, 1, 2],
    }
    result = rrelu(
        x,
        lower,
        upper,
        training=training,
        seed=None,
        rng=tf.random.Generator.from_seed(SEED),
    )
    if training:
        expect_result = training_results.get(dtype)
    else:
        expect_result = [-0.30000001192092896, -0.15000000596046448, 0, 1, 2]
    test_utils.assert_allclose_according_to_type(result, expect_result)
