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


SEED=111111
tf.random.set_seed(SEED)
random.seed(SEED)

def _ref_rrelu(x,lower=0.125, upper=0.3333333333333333):
    if x>0:
        return x
    else:
        return random.uniform(lower,upper)*x

@test_utils.run_all_in_graph_and_eager_modes
class RreluTest(tf.test.TestCase, parameterized.TestCase):
    def test_invalid(self):
        with self.assertRaisesOpError(
                "lower must be less than or equal to upper."):  # pylint: disable=bad-continuation
            y = rrelu(tf.ones(shape=(1, 2, 3)), lower=0, upper=1)
            self.evaluate(y)

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_hardshrink(self, dtype):
        x = (np.random.rand(2, 3, 4) * 2.0 - 1.0).astype(dtype)
        self.assertAllCloseAccordingToType(rrelu(x), _ref_rrelu(x))
        self.assertAllCloseAccordingToType(
            rrelu(x, 0, 1), _ref_rrelu(x, 0, 1))

    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_gradients(self, dtype):
        x = tf.constant([-1.5, -0.5, 0.5, 1.5], dtype=dtype)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y_ref = _ref_rrelu(x)
            y = rrelu(x)
        grad_ref = tape.gradient(y_ref, x)
        grad = tape.gradient(y, x)
        self.assertAllCloseAccordingToType(grad, grad_ref)

    @parameterized.named_parameters(("float32", np.float32),
                                    ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        # Only test theoretical gradients for float32 and float64
        # because of the instability of float16 while computing jacobian
        x = tf.constant([-1.5, -0.5, 0.5, 1.5], dtype=dtype)

        theoretical, numerical = tf.test.compute_gradient(
            lambda x: rrelu(x), [x])
        self.assertAllCloseAccordingToType(theoretical, numerical, atol=1e-4)

    def test_unknown_shape(self):
        fn = rrelu.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.float32))

        for shape in [(1,), (1, 2), (1, 2, 3), (1, 2, 3, 4)]:
            x = tf.ones(shape=shape, dtype=tf.float32)
            self.assertAllClose(fn(x), rrelu(x))

    def test_serialization(self):
        ref_fn = rrelu
        config = tf.keras.activations.serialize(ref_fn)
        fn = tf.keras.activations.deserialize(config)
        self.assertEqual(fn, ref_fn)

    def test_serialization_with_layers(self):
        layer = tf.keras.layers.Dense(3, activation=rrelu)
        config = tf.keras.layers.serialize(layer)
        deserialized_layer = tf.keras.layers.deserialize(config)
        self.assertEqual(deserialized_layer.__class__.__name__,
                         layer.__class__.__name__)
        self.assertEqual(deserialized_layer.activation.__name__, "rrelu")