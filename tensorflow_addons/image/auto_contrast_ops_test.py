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
""" Test auto_contrast_ops """
import sys
import pytest
import tensorflow as tf
from tensorflow_addons.image import auto_contrast_ops
from tensorflow_addons.utils import test_utils
from absl.testing import parameterized


@test_utils.run_all_in_graph_and_eager_modes
class AutoContrastTest(tf.test.TestCase, parameterized.TestCase):
    """AutoContrastTest class to test the working of
    methods images"""

    def test_contrast(self):
        """ Method to test the auto_contrast technique on images """
        if tf.executing_eagerly():
            image = tf.constant([[1, 2], [3, 4]])
            stacked_img = tf.stack([image] * 3, 2)
            contrast_image = auto_contrast_ops.autocontrast(stacked_img)
            self.assertAllEqual(tf.shape(contrast_image), tf.shape(image))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
