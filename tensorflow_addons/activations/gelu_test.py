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
"""Tests for GELU activation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import gelu
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class GELUTest(tf.test.TestCase):
    def random_test(self):
        x = tf.constant([0.5, 1.2, -0.3], dtype=tf.float32)
        val = tf.constant([0.345714, 1.0617027, -0.11462909], 
                            dtype=tf.float32)
        act = gelu(x)
        self.assertAllClose(val, self.evaluate(act), atol=1e-5)

    def test_with_numpy(self):
        x = np.array([2.5, 0.02, -0.001])
        val = tf.constant([ 2.4849157e+00, 
                            1.0159566e-02, 
                            -4.9960107e-04], 
                            dtype=tf.float32)
        
        act = gelu(x)
        self.assertAllClose(val, self.evaluate(act), atol=1e-5)


if __name__ == '__main__':
    tf.test.main()
