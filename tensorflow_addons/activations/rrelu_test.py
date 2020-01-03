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
from tensorflow_addons.activations import rrelu
from tensorflow_addons.utils import test_utils


def _ref_rrelu(x, lower, upper):
    return tf.where(x >= 0, x, (lower + upper) * x / 2)


@test_utils.run_all_in_graph_and_eager_modes
class RreluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    @tf.function
    def test_rrelu(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        lower = 0.1
        upper = 0.2
        result = rrelu(x, lower, upper, training=False)
        expect_result = _ref_rrelu(x, lower, upper)
        self.assertAllCloseAccordingToType(result, expect_result)

    @parameterized.named_parameters(("float32", np.float32),
                                    ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        x = tf.constant([-2.0, -1.0, -0.1, 0.1, 1.0, 2.0], dtype=dtype)
        lower = 0.1
        upper = 0.2
        for training in [True, False]:
            with self.subTest(training=training):
                theoretical, numerical = tf.test.compute_gradient(
                    lambda x: rrelu(
                        x, lower, upper, training=training, seed=111111), [x])
                # TODO: investigate the difference between CPU and GPU
                if training is True and tf.test.is_gpu_available() is False:
                    numerical = [[[0.134971, 0., 0., 0., 0., 0.],
                                  [0., 0.15648358, 0., 0., 0., 0.],
                                  [0., 0., 0.18776372, 0., 0., 0.],
                                  [0., 0., 0., 1., 0., 0.],
                                  [0., 0., 0., 0., 1., 0.],
                                  [0., 0., 0., 0., 0., 1.]]]
                self.assertAllCloseAccordingToType(
                    theoretical, numerical, rtol=5e-4, atol=5e-4)


if __name__ == "__main__":
    tf.test.main()
