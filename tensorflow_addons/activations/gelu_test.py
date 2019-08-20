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

import tensorflow as tf
from absl.testing import parameterized
from tensorflow_addons.activations import gelu
from tensorflow_addons.utils import test_utils



@parameterized.parameters(['float16', 'float32', 'float64'])
@test_utils.run_all_in_graph_and_eager_modes
class TestGelu(tf.test.TestCase):
    def test_random(self, dtype):
        x = tf.constant([0.5, 1.2, -0.3], dtype=dtype)
        val = tf.constant([0.345714, 1.0617027, -0.11462909], dtype=dtype)
        act = gelu(x, dtype)
        self.assertAllClose(val, self.evaluate(act), atol=1e-5)

if __name__ == '__main__':
    tf.test.main()
