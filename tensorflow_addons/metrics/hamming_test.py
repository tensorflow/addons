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
"""Tests Hamming metrics."""

import tensorflow as tf
from tensorflow_addons.metrics import HammingLoss, hamming_distance
from tensorflow_addons.utils import test_utils
from tensorflow.keras import layers
import numpy as np


@test_utils.run_all_in_graph_and_eager_modes
class HammingMetricsTest(tf.test.TestCase):
    def test_config(self):
        hl_obj = HammingLoss(mode='multilabel', threshold=0.8)
        self.assertEqual(hl_obj.name, 'hamming_loss')
        self.assertEqual(hl_obj.dtype, tf.float32)

    def initialize_vars(self, mode, threshold):
        hl_obj = HammingLoss(mode=mode, threshold=threshold)
        return hl_obj

    def update_obj_states(self, obj, actuals, preds):
        update_op = obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_mc_4_classes(self):
        actuals = tf.constant(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0],
             [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]],
            dtype=tf.float32)
        predictions = tf.constant(
            [[0.85, 0.12, 0.03, 0], [0, 0, 1, 0], [0.10, 0.045, 0.045, 0.81],
             [1, 0, 0, 0], [0.80, 0.10, 0.10, 0], [1, 0, 0, 0],
             [0.05, 0, 0.90, 0.05]],
            dtype=tf.float32)
        # Initialize
        hl_obj = self.initialize_vars('multiclass', 0.8)
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.2857143)

    def test_mc_5_classes(self):
        actuals = tf.constant(
            [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=tf.float32)

        predictions = tf.constant(
            [[0.85, 0, 0.15, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0],
             [0.05, 0.90, 0.04, 0, 0.01], [0.10, 0, 0.81, 0.09, 0],
             [0.10, 0.045, 0, 0.81, 0.045], [1, 0, 0, 0, 0],
             [0, 0.85, 0, 0, 0.15]],
            dtype=tf.float32)
        # Initialize
        hl_obj = self.initialize_vars('multiclass', 0.8)
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.25)

    def test_ml_4_classes(self):
        actuals = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]],
                              dtype=tf.float32)
        predictions = tf.constant(
            [[0.97, 0.56, 0.83, 0.77], [0.34, 0.95, 0.7, 0.89],
             [0.95, 0.45, 0.23, 0.56]],
            dtype=tf.float32)
        # Initialize
        hl_obj = self.initialize_vars('multilabel', 0.8)
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.16666667)

    def test_ml_5_classes(self):
        actuals = tf.constant(
            [[1, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0],
             [1, 0, 0, 0, 1], [0, 1, 1, 0, 0]],
            dtype=tf.float32)
        predictions = tf.constant(
            [[1, 0.75, 0.2, 0.55, 0], [0.65, 0.22, 0.97, 0.88, 0],
             [0, 1, 0, 1, 0], [0, 0.85, 0.9, 0.34, 0.5],
             [0.4, 0.65, 0.87, 0, 0.12], [0.66, 0.55, 1, 0.98, 0],
             [0.95, 0.34, 0.67, 0.65, 0.10], [0.45, 0.97, 0.89, 0.67, 0.46]],
            dtype=tf.float32)
        # Initialize
        hl_obj = self.initialize_vars('multilabel', 0.7)
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.075)

    def hamming_distance_test(self):
        actuals = tf.constant([1, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=tf.int32)
        predictions = tf.constant([1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                                  dtype=tf.int32)
        test_result = hamming_distance(actuals, predictions)
        self.assertAllClose(0.3, test_result, atol=1e-5)

    # Keras model check
    def test_keras_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))
        h1 = HammingLoss(mode='multiclass')
        model.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=[h1])
        data = np.random.random((100, 10))
        labels = np.random.random((100, 3))
        model.fit(data, labels, epochs=1, batch_size=32, verbose=0)
