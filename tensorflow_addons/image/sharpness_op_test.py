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
"""Tests for shapness"""

import pytest
import tensorflow as tf
import numpy as np

from tensorflow_addons.image import sharpness_op
from PIL import Image, ImageEnhance

_DTYPES = {
    np.uint8,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
}


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_equalize_with_PIL():
    # np.random.seed(0)
    image = np.random.randint(low=0, high=255, size=(5, 5, 3), dtype=np.uint8)
    enhancer = ImageEnhance.Sharpness(Image.fromarray(image))
    sharpened = enhancer.enhance(0.5)
    np.testing.assert_allclose(
        sharpness_op.sharpness(tf.constant(image), 0.5).numpy(), sharpened, atol=1
    )
