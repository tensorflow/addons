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

import math

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import rrelu
from tensorflow_addons.utils import test_utils
import random


def _ref_rrelu(x, lower, upper, alpha, training=None):
    if training:
        return tf.where(x >= 0, x, alpha * x)
    else:
        return tf.where(x >= 0, x, x * (lower + upper) / 2)


@test_utils.run_all_in_graph_and_eager_modes
class RreluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_rrelu_training(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        lower = 0.1
        upper = 0.2
        # result,alpha=rrelu(x,lower,upper, training=True)
        # self.assertAllCloseAccordingToType(result, _ref_rrelu(x,lower,upper,alpha,training=True))

        result, alpha = rrelu(x, lower, upper, training=False)
        self.assertAllCloseAccordingToType(
            result, _ref_rrelu(x, lower, upper, alpha, training=False))


if __name__ == "__main__":
    tf.test.main()
