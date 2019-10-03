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
"""Tests for Multilabel Confusion Matrix Metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class MultiLabelConfusionMatrixTest(tf.test.TestCase):
    def test_config(self):
        mcm_obj = MultiLabelConfusionMatrix(num_classes=3, threshold=0.9)
        self.assertEqual(mcm_obj.num_classes, 3)
        self.assertEqual(mcm_obj.threshold, 0.9)
        # Check save and restore config
        mcm_obj2 = MultiLabelConfusionMatrix.from_config(mcm_obj.get_config())
        self.assertEqual(mcm_obj2.num_classes, 3)
        self.assertEqual(mcm_obj2.threshold, 0.9)

    def initialize_vars(self, n_classes, threshold):
        mcm_obj = MultiLabelConfusionMatrix(
            num_classes=n_classes, threshold=threshold)
        self.evaluate(tf.compat.v1.variables_initializer(mcm_obj.variables))
        return mcm_obj

    def update_obj_states(self, obj, actuals, preds):
        update_op = obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_mcm_3_classes(self):
        actuals = tf.constant([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
                              dtype=tf.float32)
        preds = tf.constant([[0.90, 0.5, 0.78], [0.23, 0.899, 1],
                             [0.921, 0.323, 0.546], [0.123, 0.962, 1]],
                            dtype=tf.float32)
        # Initialize
        mcm_obj = self.initialize_vars(n_classes=3, threshold=0.85)
        # Update
        self.update_obj_states(mcm_obj, actuals, preds)
        # Check results
        self.check_results(
            mcm_obj, [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[0, 2], [2, 0]]])

    def test_mcm_4_classes(self):
        actuals = tf.constant(
            [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0],
             [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
             [0, 1, 1, 0], [0, 1, 0, 1]],
            dtype=tf.float32)
        preds = tf.constant(
            [[0.71, 0.69, 1, 0.545], [0.326, 0.2323, 1, 0.921],
             [0.243, 0, 0.343, 0.891], [1, 0.967, 0.232, 0.656],
             [0.901, 0.32, 0.5, 0.1], [0.91, 0.02, 0.54, 1.0],
             [0, 0.612, 0.812, 0.79], [0.816, 0.12, 0.34, 0.821],
             [0.1365, 0.921, 0.54, 0.52], [0.12, 0.64, 0.43, 0.97]],
            dtype=tf.float32)
        # Initialize
        mcm_obj = self.initialize_vars(n_classes=4, threshold=0.70)
        # Update
        self.update_obj_states(mcm_obj, actuals, preds)
        # Check results
        self.check_results(mcm_obj, [[[4, 1], [1, 4]], [[6, 0], [2, 2]],
                                     [[6, 1], [1, 2]], [[2, 0], [2, 6]]])

    def test_multiclass(self):
        actuals = tf.constant(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0],
             [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
             [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=tf.float32)
        preds = tf.constant(
            [[0.81, 0.23, 0.456, 0.54], [0.232, 0.46, 0.90, 0.23],
             [0.12, 0.57, 0.65, 0.912], [1, 0.232, 0.46, 0.79],
             [0.85, 0.76, 0.12, 0.55], [0.81, 0.45, 0.66, 0.78],
             [0.34, 0.67, 0.81, 0.45], [0.8923, 0.23, 0.56, 0.67],
             [0.12, 0.98, 0.23, 0.6], [0.34, 0.5, 0.7, 1]],
            dtype=tf.float32)
        # Initialize
        mcm_obj = self.initialize_vars(n_classes=4, threshold=0.8)
        # Update
        self.update_obj_states(mcm_obj, actuals, preds)
        # Check results
        self.check_results(mcm_obj, [[[5, 2], [0, 3]], [[7, 1], [2, 0]],
                                     [[7, 0], [1, 2]], [[8, 0], [0, 2]]])
