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

from tensorflow_addons.image import augment_ops
from tensorflow_addons.utils import test_utils

_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


@test_utils.run_all_in_graph_and_eager_modes
class AugemntOpTest(tf.test.TestCase):
    def test_augment(self):
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
            blended = augment_ops.blend(image1, image2, 0.5)
            self.assertAllEqual(
                self.evaluate(blended),
                [
                    [127, 127, 127, 127],
                    [127, 127, 127, 127],
                    [127, 127, 127, 127],
                    [127, 127, 127, 127],
                ],
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
