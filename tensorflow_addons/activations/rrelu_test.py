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


def _ref_rrelu(x, alpha):
    return tf.where(x >= 0, x, alpha * x)


def _ref_rrelu_grad(x, alpha, dtype):
    return tf.where(x >= 0, tf.constant(1, dtype=dtype), alpha)


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
        result, alpha = rrelu(x, lower, upper, training=True, with_alpha=True)
        expect_result = _ref_rrelu(x, alpha)
        self.assertAllCloseAccordingToType(result, expect_result)

        result, alpha = rrelu(x, lower, upper, training=False, with_alpha=True)
        expect_result = _ref_rrelu(x, alpha)
        self.assertAllCloseAccordingToType(result, expect_result)

    @parameterized.named_parameters(("float32", np.float32),
                                    ("float64", np.float64))
    @tf.function
    def test_theoretical_gradients(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        lower = 0.1
        upper = 0.2
        with tf.GradientTape() as t:
            t.watch(x)
            result, alpha = rrelu(
                x, lower, upper, training=True, with_alpha=True)
        grad = t.gradient(result, x)
        expect_grad = _ref_rrelu_grad(x, alpha, dtype)
        self.assertAllCloseAccordingToType(grad, expect_grad, atol=1e-4)

    def test_unknown_shape(self):
        fn = rrelu.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.float32))

        for shape in [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]:
            x = tf.ones(shape=shape, dtype=tf.float32)
            self.assertAllClose(fn(x), rrelu(x))


if __name__ == "__main__":
    tf.test.main()
