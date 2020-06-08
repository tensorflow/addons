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
"""Tests for GELU activation."""


import pytest
import numpy as np
from tensorflow_addons.layers.gelu import GELU
from tensorflow_addons.utils import test_utils


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_random(dtype):
    x = np.array([[0.5, 1.2, -0.3]]).astype(dtype)
    val = np.array([[0.345714, 1.0617027, -0.11462909]]).astype(dtype)
    test_utils.layer_test(
        GELU, kwargs={"dtype": dtype}, input_data=x, expected_output=val
    )
