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
"""Tests of color ops"""

import pytest
import tensorflow as tf
import numpy as np

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
@pytest.mark.parametrize("shape", [(4, 4,), (4, 4, 1), (4, 4, 3), (5, 4, 4, 3)])
def test_equalize_dtype_shape(dtype, shape):
    image = np.ones(shape=shape, dtype=dtype)
    equalized = color_ops.equalize(tf.constant(image)).numpy()
    np.testing.assert_equal(equalized, image)
    assert equalized.dtype == image.dtype


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_equalize_with_PIL():
    image = np.random.randint(low=0, high=255, size=(4, 3, 3, 3), dtype=np.uint8)
    batches = []
    for i in range(4):
        batch = ImageOps.equalize(Image.fromarray(image[i, :]))
        batches.append(batch)
    equalized = np.stack(batches)
    np.testing.assert_equal(color_ops.equalize(tf.constant(image)).numpy(), equalized)
