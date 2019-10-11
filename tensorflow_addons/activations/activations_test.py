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

import tensorflow as tf
from tensorflow_addons import activations
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class ActivationsTest(tf.test.TestCase):

    ALL_ACTIVATIONS = [
        "gelu", "hardshrink", "lisht", "mish", "softshrink", "sparsemax",
        "tanhshrink"
    ]

    def test_serialization(self):
        for name in self.ALL_ACTIVATIONS:
            fn = tf.keras.activations.get(name)
            ref_fn = getattr(activations, name)
            self.assertEqual(fn, ref_fn)
            config = tf.keras.activations.serialize(fn)
            fn = tf.keras.activations.deserialize(config)
            self.assertEqual(fn, ref_fn)

    def test_serialization_with_layers(self):
        for name in self.ALL_ACTIVATIONS:
            layer = tf.keras.layers.Dense(
                3, activation=getattr(activations, name))
            config = tf.keras.layers.serialize(layer)
            deserialized_layer = tf.keras.layers.deserialize(config)
            self.assertEqual(deserialized_layer.__class__.__name__,
                             layer.__class__.__name__)
            self.assertEqual(deserialized_layer.activation.__name__, name)
