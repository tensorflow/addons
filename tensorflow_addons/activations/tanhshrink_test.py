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
from tensorflow_addons.activations import tanhshrink
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class TanhshrinkTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_tanhshrink(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        expected_result = tf.constant(
            [-1.0359724, -0.23840582, 0.0, 0.23840582, 1.0359724], dtype=dtype)

        self.assertAllCloseAccordingToType(tanhshrink(x), expected_result)

    @parameterized.named_parameters(("float32", np.float32),
                                    ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        # Only test theoretical gradients for float32 and float64
        # because of the instability of float16 while computing jacobian
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)

        theoretical, numerical = tf.test.compute_gradient(tanhshrink, [x])
        self.assertAllCloseAccordingToType(theoretical, numerical, atol=1e-4)

    def test_serialization(self):
        config = tf.keras.activations.serialize(tanhshrink)
        fn = tf.keras.activations.deserialize(config)
        self.assertEqual(fn, tanhshrink)

    def test_serialization_with_layers(self):
        layer = tf.keras.layers.Dense(3, activation=tanhshrink)
        config = tf.keras.layers.serialize(layer)
        deserialized_layer = tf.keras.layers.deserialize(config)
        self.assertEqual(deserialized_layer.__class__.__name__,
                         layer.__class__.__name__)
        self.assertEqual(deserialized_layer.activation.__name__, "tanhshrink")


if __name__ == "__main__":
    tf.test.main()
