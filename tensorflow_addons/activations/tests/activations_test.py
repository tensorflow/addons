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

import pytest
import tensorflow as tf
from tensorflow_addons import activations


ALL_ACTIVATIONS = [
    "gelu",
    "hardshrink",
    "lisht",
    "mish",
    "rrelu",
    "softshrink",
    "sparsemax",
    "tanhshrink",
    "snake",
]


@pytest.mark.parametrize("name", ALL_ACTIVATIONS)
def test_serialization(name):
    fn = tf.keras.activations.get("Addons>" + name)
    ref_fn = getattr(activations, name)
    assert fn == ref_fn
    config = tf.keras.activations.serialize(fn)
    fn = tf.keras.activations.deserialize(config)
    assert fn == ref_fn


@pytest.mark.parametrize("name", ALL_ACTIVATIONS)
def test_serialization_with_layers(name):
    layer = tf.keras.layers.Dense(3, activation=getattr(activations, name))
    config = tf.keras.layers.serialize(layer)
    deserialized_layer = tf.keras.layers.deserialize(config)
    assert deserialized_layer.__class__.__name__ == layer.__class__.__name__
    assert deserialized_layer.activation.__name__ == name
