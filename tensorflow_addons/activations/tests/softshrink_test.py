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

import pytest

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import softshrink

from tensorflow_addons.utils import test_utils


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_softshrink(dtype):
    x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
    expected_result = tf.constant([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=dtype)
    test_utils.assert_allclose_according_to_type(softshrink(x), expected_result)

    expected_result = tf.constant([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype)
    test_utils.assert_allclose_according_to_type(
        softshrink(x, lower=-1.0, upper=1.0), expected_result
    )
