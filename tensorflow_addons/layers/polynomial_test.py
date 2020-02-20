# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for PolynomialCrossing layer."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.polynomial import PolynomialCrossing
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class PolynomialCrossingTest(tf.test.TestCase):
    # Do not use layer_test due to multiple inputs.

    def test_full_matrix(self):
        x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
        x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
        layer = PolynomialCrossing(projection_dim=None, kernel_initializer="ones")
        output = layer([x0, x])
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(np.asarray([[0.55, 0.8, 1.05]]), output)

    def test_invalid_proj_dim(self):
        with self.assertRaisesRegexp(ValueError, r"is not supported yet"):
            x0 = np.random.random((12, 5))
            x = np.random.random((12, 5))
            layer = PolynomialCrossing(projection_dim=6)
            layer([x0, x])

    def test_invalid_inputs(self):
        with self.assertRaisesRegexp(ValueError, r"must be a tuple or list of size 2"):
            x0 = np.random.random((12, 5))
            x = np.random.random((12, 5))
            x1 = np.random.random((12, 5))
            layer = PolynomialCrossing(projection_dim=6)
            layer([x0, x, x1])

    def test_serialization(self):
        layer = PolynomialCrossing(projection_dim=None)
        serialized_layer = tf.keras.layers.serialize(layer)
        new_layer = tf.keras.layers.deserialize(serialized_layer)
        self.assertEqual(layer.get_config(), new_layer.get_config())


if __name__ == "__main__":
    tf.test.main()
