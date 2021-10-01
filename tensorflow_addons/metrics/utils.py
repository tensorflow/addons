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
"""Utilities for metrics."""

import numpy as np
import tensorflow as tf


def _get_model(metric, num_output):
    # Test API comptibility with tf.keras Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_output, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["acc", metric]
    )

    data = np.random.random((10, 3))
    labels = np.random.random((10, num_output))
    model.fit(data, labels, epochs=1, batch_size=5, verbose=0)


def sample_weight_shape_match(v, sample_weight):
    if sample_weight is None:
        return tf.ones_like(v)
    if np.size(sample_weight) == 1:
        return tf.fill(v.shape, sample_weight)
    return tf.convert_to_tensor(sample_weight)
