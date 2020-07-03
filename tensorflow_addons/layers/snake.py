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
"""Implements Snake layer."""

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
class Snake(tf.keras.layers.Layer):
    """Snake layer to learn periodic functions with the 'frequency' param trainable.

    https://arxiv.org/abs/2006.08195
    """

    @typechecked
    def __init__(self, freq_initializer: types.Initializer = "ones", **kwargs):
        """
        freq_initializer: Initializer for the 'frequency' param.
        """
        super().__init__(**kwargs)
        self.freq_initializer = tf.keras.initializers.get(freq_initializer)
        self.freq = self.add_weight(initializer=freq_initializer, trainable=True)

    def call(self, inputs):
        return inputs + (1 - tf.cos(2 * self.freq * inputs)) / (2 * self.freq)

    def get_config(self):
        config = {
            "freq_initializer": tf.keras.initializers.serialize(self.freq_initializer),
        }
        base_config = super().get_config()
        return {**base_config, **config}
