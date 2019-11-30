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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import swish
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class SwishTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_swish(self, dtype):
        x = tf.constant([-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0], dtype=dtype)
        expected_result = tf.constant([
                -0.000453978687024,
                -0.268941421369995,
                -0.188770334399073,
                0.0,
                0.311229665600927,
                0.731058578630005,
                9.99954602131298,
                ], dtype=dtype)
        self.assertAllCloseAccordingToType(swish(x), expected_result)


if __name__ == "__main__":
    tf.test.main()
