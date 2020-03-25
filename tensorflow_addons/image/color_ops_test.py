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
"""Tests colour ops"""

import tensorflow as tf
import numpy as np
import pytest

from tensorflow_addons.image import color_ops
from PIL import Image, ImageOps

_DTYPES = {
    np.uint8,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
}


@pytest.mark.parametrize("dtype", _DTYPES)
def test_invert_dtype(dtype):
    image = np.ones(shape=(3, 3, 3), dtype=dtype) * 127
    inverted = color_ops.invert(tf.constant(image, dtype))
    np.testing.assert_equal(np.ones((3, 3, 3), dtype=dtype) * 128, inverted)
    assert inverted.numpy().dtype == dtype


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 3)])
def test_invert_with_PIL(shape):
    image = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)
    np.testing.assert_equal(
        ImageOps.invert(Image.fromarray(image)),
        color_ops.invert(tf.constant(image)).numpy(),
    )
