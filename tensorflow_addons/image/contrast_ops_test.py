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
"""Tests for contrast ops."""

import sys

import pytest
import tensorflow as tf
import numpy as np
from absl.testing import parameterized
from tensorflow_addons.image.contrast_ops import autocontrast


@parameterized.named_parameters(
    ("float16", np.float16), ("float32", np.float32), ("uint8", np.uint8)
)
def test_different_dtypes(dtype):
    test_image = tf.ones([1, 40, 40, 3], dtype=dtype)
    result_image = autocontrast(test_image)
    np.testing.assert_allclose(result_image, test_image)


def test_different_channels():
    for channel in [1, 3, 4]:
        test_image = tf.ones([1, 40, 40, channel], dtype=np.uint8)
        result_image = autocontrast(test_image)
        np.testing.assert_allclose(result_image, test_image)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
