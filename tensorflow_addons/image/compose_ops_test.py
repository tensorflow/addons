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

from tensorflow_addons.image import compose_ops
from tensorflow_addons.utils import test_utils

_DTYPES = {
    tf.dtypes.uint8,
}


@test_utils.run_all_in_graph_and_eager_modes
class ComposeOpTest(tf.test.TestCase):
    def test_blend(self):
        for dtype in _DTYPES:
            image1_file = tf.io.read_file(
                "tensorflow_addons/image/test_data/Yellow_Smiley_Face.png"
            )
            test_image_1 = tf.io.decode_image(image1_file, channels=3, dtype=tf.uint8)
            image2_file = tf.io.read_file(
                "tensorflow_addons/image/test_data/Yellow_Smiley_Face_Warp-interp-3-clamp-4.png"
            )
            test_image_2 = tf.io.decode_image(image2_file, channels=3, dtype=tf.uint8)
            result_image = compose_ops.blend(test_image_1, test_image_2, 0.5)
            blended_file = tf.io.read_file(
                "tensorflow_addons/image/test_data/Yellow_Smiley_Face_Blend-Half.png"
            )
            blended_image = tf.io.decode_image(blended_file)
            self.assertAllEqual(result_image, blended_image)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
