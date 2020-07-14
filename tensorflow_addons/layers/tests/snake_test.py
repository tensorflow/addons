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
"""Tests for Snake layer."""

import pytest

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.snake import Snake
from tensorflow_addons.activations.snake import snake

from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_layer(dtype):
    x = np.random.rand(2, 5).astype(dtype)
    a = np.random.randn()
    val = snake(x, a)
    test_utils.layer_test(
        Snake,
        kwargs={"frequency_initializer": tf.constant_initializer(a), "dtype": dtype},
        input_data=x,
        expected_output=val,
    )
