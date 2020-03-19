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
"""Test of solarize_ops"""

import sys
import pytest
import tensorflow as tf
from tensorflow_addons.image import solarize_ops
from tensorflow_addons.utils import test_utils
from absl.testing import parameterized


@test_utils.run_all_in_graph_and_eager_modes
class SolarizeOPSTest(tf.test.TestCase, parameterized.TestCase):
    """SolarizeOPSTest class to test the solarize images"""

    def test_solarize(self):
        if tf.executing_eagerly():
            image2 = tf.constant(
                [
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                ],
                dtype=tf.uint8,
            )
            threshold = 10
            solarize_img = solarize_ops.solarize(image2, threshold)
            self.assertAllEqual(tf.shape(solarize_img), tf.shape(image2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
