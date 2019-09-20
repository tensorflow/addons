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


def _ref_tanhshrink(x):
    return x - tf.tanh(x)


@test_utils.run_all_in_graph_and_eager_modes
class TanhshrinkTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_tanhshrink(self, dtype):
        x = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
        self.assertAllCloseAccordingToType(tanhshrink(x), _ref_tanhshrink(x))

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_gradients(self, dtype):
        x = tf.constant([1.0, 2.0, 3.0], dtype=dtype)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_ref = _ref_tanhshrink(x)
            y = tanhshrink(x)
        grad_ref = tape.gradient(y_ref, x)
        grad = tape.gradient(y, x)
        self.assertAllCloseAccordingToType(grad, grad_ref)

    def test_serialization(self):
        ref_fn = tanhshrink
        config = tf.keras.activations.serialize(ref_fn)
        fn = tf.keras.activations.deserialize(config)
        self.assertEqual(fn, ref_fn)


if __name__ == "__main__":
    tf.test.main()
