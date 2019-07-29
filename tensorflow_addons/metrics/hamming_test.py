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
"""Tests F1 metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import HammingLoss, hamming_distance
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class HammingMetricsTest(tf.test.TestCase):
    def test_config(self):
        hl_obj = HammingLoss(mode='multilabel')
        self.assertEqual(hl_obj.name, 'hamming_loss')
        self.assertEqual(hl_obj.dtype, tf.float32)

    def initialize_vars(self, mode):
        hl_obj = HammingLoss(mode=mode)
        return hl_obj

    def update_obj_states(self, obj, actuals, preds):
        update_op = obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_mc_4_classes(self):
        actuals = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0],
                               [0, 0, 0, 1], [0, 1, 0, 0],
                               [0, 1, 0, 0], [1, 0, 0, 0],
                               [0, 0, 1, 0]], dtype=tf.float32)
        predictions = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0],
                                   [0, 0, 0, 1], [1, 0, 0, 0],
                                   [1, 0, 0, 0], [1, 0, 0, 0],
                                   [0, 0, 1, 0]], dtype=tf.float32)
        # Initialize
        hl_obj = self.initialize_vars('multiclass')
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.2857143)

    def test_mc_5_classes(self):
        actuals = tf.constant([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1], [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0], [0, 0, 1, 0, 0],
                               [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
                              dtype=tf.int32)
        predictions = tf.constant([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0],
                                   [0, 1, 0, 0, 0],
                                   [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                   [0, 0, 0, 1, 0], [1, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0]], dtype=tf.int32)
        # Initialize
        hl_obj = self.initialize_vars('multiclass')
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.25)

    def test_ml_4_classes(self):
        actuals = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1],
                               [0, 0, 0, 1]], dtype=tf.float32)
        predictions = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1],
                                   [1, 0, 0, 0]], dtype=tf.float32)
        # Initialize
        hl_obj = self.initialize_vars('multilabel')
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.16666667)

    def test_ml_5_classes(self):
        actuals = tf.constant([[1, 0, 0, 0, 0], [0, 0, 1, 1, 0],
                               [0, 1, 0, 1, 0], [0, 1, 1, 0, 0],
                               [0, 0, 1, 1, 0], [0, 0, 1, 1, 0],
                               [1, 0, 0, 0, 1], [0, 1, 1, 0, 0]],
                              dtype=tf.int32)
        predictions = tf.constant([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0],
                                   [0, 1, 0, 1, 0], [0, 1, 1, 0, 0],
                                   [0, 0, 1, 0, 0], [0, 0, 1, 1, 0],
                                   [1, 0, 0, 0, 0], [0, 1, 1, 0, 0]],
                                  dtype=tf.int32)
        # Initialize
        hl_obj = self.initialize_vars('multilabel')
        # Update
        self.update_obj_states(hl_obj, actuals, predictions)
        # Check results
        self.check_results(hl_obj, 0.075)

    def hamming_distance_test(self):
        actuals = tf.constant([1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                              dtype=tf.int32)
        predictions = tf.constant([1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                                  dtype=tf.int32)
        test_result = hamming_distance(actuals, predictions)
        self.assertAllClose(0.3, test_result, atol=1e-5)
