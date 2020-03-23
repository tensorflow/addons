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

import sys

import pytest
from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import gelu
from tensorflow_addons.activations.gelu import _gelu_py
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class GeluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("float16", np.float16), ("float32", np.float32), ("float64", np.float64)
    )
    def test_gelu(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        expected_result = tf.constant(
            [-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977], dtype=dtype
        )
        self.assertAllCloseAccordingToType(gelu(x), expected_result)

        expected_result = tf.constant(
            [-0.04550028, -0.15865526, 0.0, 0.8413447, 1.9544997], dtype=dtype
        )
        self.assertAllCloseAccordingToType(gelu(x, False), expected_result)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        # Only test theoretical gradients for float32 and float64
        # because of the instability of float16 while computing jacobian
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)

        for approximate in [True, False]:
            with self.subTest(approximate=approximate):
                theoretical, numerical = tf.test.compute_gradient(
                    lambda x: gelu(x, approximate=approximate), [x]
                )
                self.assertAllCloseAccordingToType(theoretical, numerical, atol=1e-4)

    @parameterized.named_parameters(("float32", np.float32), ("float64", np.float64))
    def test_same_as_py_func(self, dtype):
        np.random.seed(100)
        for _ in range(20):
            self.verify_funcs_are_equivalent(dtype)

    def verify_funcs_are_equivalent(self, dtype):
        x_np = np.random.uniform(-10, 10, size=(4, 4)).astype(dtype)
        x = tf.convert_to_tensor(x_np)
        for approximate in [True, False]:
            with tf.GradientTape(persistent=True) as t:
                t.watch(x)
                y_native = gelu(x, approximate=approximate)
                y_py = _gelu_py(x, approximate=approximate)
            self.assertAllCloseAccordingToType(y_native, y_py)
            grad_native = t.gradient(y_native, x)
            grad_py = t.gradient(y_py, x)
            # TODO: lower atol to 1e-6
            # currently it doesn't work.
            # It necessitates changing the Python or C++ implementation.
            self.assertAllCloseAccordingToType(grad_native, grad_py, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
