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

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import softshrink
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class SoftshrinkTest(tf.test.TestCase, parameterized.TestCase):
    def test_invalid(self):
        with self.assertRaisesOpError(
            "lower must be less than or equal to upper."
        ):  # pylint: disable=bad-continuation
            y = softshrink(tf.ones(shape=(1, 2, 3)), lower=2.0, upper=-2.0)
            self.evaluate(y)

    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_softshrink(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        expected_result = tf.constant([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=dtype)
        self.assertAllCloseAccordingToType(softshrink(x), expected_result)

        expected_result = tf.constant([-1.0, 0.0, 0.0, 0.0, 1.0], dtype=dtype)
        self.assertAllCloseAccordingToType(
            softshrink(x, lower=-1.0, upper=1.0), expected_result
        )

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        # Only test theoretical gradients for float32 and float64
        # because of the instability of float16 while computing jacobian

        # Softshrink is not continuous at `lower` and `upper`.
        # Avoid these two points to make gradients smooth.
        x = tf.constant([-2.0, -1.5, 0.0, 1.5, 2.0], dtype=dtype)

        theoretical, numerical = tf.test.compute_gradient(softshrink, [x])
        self.assertAllCloseAccordingToType(theoretical, numerical, atol=1e-4)


if __name__ == "__main__":
    tf.test.main()
