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
from tensorflow_addons.activations import gelu
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class GeluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_gelu(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        expected_result = tf.constant(
            [-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977], dtype=dtype)
        self.assertAllCloseAccordingToType(gelu(x), expected_result)

        expected_result = tf.constant(
            [-0.04550028, -0.15865526, 0.0, 0.8413447, 1.9544997], dtype=dtype)
        self.assertAllCloseAccordingToType(gelu(x, False), expected_result)

    @parameterized.named_parameters(("float32", np.float32),
                                    ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        # Only test theoretical gradients for float32 and float64
        # because of the instability of float16 while computing jacobian
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)

        for approximate in [True, False]:
            with self.subTest(approximate=approximate):
                theoretical, numerical = tf.test.compute_gradient(
                    lambda x: gelu(x, approximate=approximate), [x])
                self.assertAllCloseAccordingToType(
                    theoretical, numerical, atol=1e-4)

    def test_unknown_shape(self):
        fn = gelu.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.float32))

        for shape in [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]:
            x = tf.ones(shape=shape, dtype=tf.float32)
            self.assertAllClose(fn(x), gelu(x))

    def test_serialization(self):
        config = tf.keras.activations.serialize(gelu)
        fn = tf.keras.activations.deserialize(config)
        self.assertEqual(fn, gelu)

    def test_serialization_with_layers(self):
        layer = tf.keras.layers.Dense(3, activation=gelu)
        config = tf.keras.layers.serialize(layer)
        deserialized_layer = tf.keras.layers.deserialize(config)
        self.assertEqual(deserialized_layer.__class__.__name__,
                         layer.__class__.__name__)
        self.assertEqual(deserialized_layer.activation.__name__, "gelu")


if __name__ == "__main__":
    tf.test.main()
