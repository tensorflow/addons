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
"""Tests of augmentation ops"""

import pytest
import tensorflow as tf
import numpy as np

from tensorflow_addons.image import compose_ops

_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


def blend_np(image1, image2, factor):
    image1 = image1.astype("float32")
    image2 = image2.astype("float32")
    difference = image2 - image1
    scaled = factor * difference
    temp = image1 + scaled
    if factor >= 0.0 and factor <= 1.0:
        temp = np.round(temp)
        return temp
    temp = np.round(np.clip(temp, 0.0, 255.0))
    return temp


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_blend(dtype):
    image1 = tf.constant(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=dtype
    )
    image2 = tf.constant(
        [
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
            [255, 255, 255, 255],
        ],
        dtype=dtype,
    )
    blended = compose_ops.blend(image1, image2, 0.5).numpy()
    np.testing.assert_equal(
        blended,
        [
            [128, 128, 128, 128],
            [128, 128, 128, 128],
            [128, 128, 128, 128],
            [128, 128, 128, 128],
        ],
    )

    np.random.seed(0)
    image1 = np.random.randint(0, 255, (3, 5, 5), np.uint8)
    image2 = np.random.randint(0, 255, (3, 5, 5), np.uint8)
    tf.random.set_seed(0)
    factor = tf.random.uniform(shape=[], maxval=1, dtype=tf.dtypes.float32, seed=0)
    blended = compose_ops.blend(
        tf.convert_to_tensor(image1), tf.convert_to_tensor(image2), factor
    ).numpy()
    expected = blend_np(image1, image2, factor.numpy())
    np.testing.assert_equal(blended, expected)
    assert blended.dtype == expected.dtype
