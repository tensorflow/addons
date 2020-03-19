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

import sys
import pytest
import tensorflow as tf
import numpy as np

from tensorflow_addons.image import compose_ops
from tensorflow_addons.utils import test_utils

_DTYPES = {
    tf.dtypes.uint8,
}


def blend_np(image1, image2, factor):
    image1 = image1.astype("float32")
    image2 = image2.astype("float32")
    difference = image2 - image1
    scaled = factor * difference
    temp = image1 + scaled
    if factor >= 0.0 and factor <= 1.0:
        temp = np.round(temp)
        return temp.astype("uint8")
    temp = np.round(np.clip(temp, 0.0, 255.0))
    return temp.astype("uint8")


@test_utils.run_all_in_graph_and_eager_modes
class ComposeOpTest(tf.test.TestCase):
    def test_blend(self):
        for dtype in _DTYPES:
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
            blended = compose_ops.blend(image1, image2, 0.5)
            self.assertAllEqual(
                self.evaluate(blended),
                [
                    [128, 128, 128, 128],
                    [128, 128, 128, 128],
                    [128, 128, 128, 128],
                    [128, 128, 128, 128],
                ],
            )

            image1 = np.random.randint(0, 255, (4, 4, 3), np.uint8)
            image2 = np.random.randint(0, 255, (4, 4, 3), np.uint8)
            blended = compose_ops.blend(
                tf.convert_to_tensor(image1), tf.convert_to_tensor(image2), 0.35
            )
            self.assertAllEqual(self.evaluate(blended), blend_np(image1, image2, 0.35))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
