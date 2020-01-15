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

import tensorflow as tf
from tensorflow_addons.metrics import MultiLabelConfusionMatrix
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class MultiLabelConfusionMatrixTest(tf.test.TestCase):
    def test_config(self):
        mcm_obj = MultiLabelConfusionMatrix(num_classes=3)
        self.assertEqual(mcm_obj.num_classes, 3)
        self.assertEqual(mcm_obj.dtype, tf.int32)
        # Check save and restore config
        mcm_obj2 = MultiLabelConfusionMatrix.from_config(mcm_obj.get_config())
        self.assertEqual(mcm_obj2.num_classes, 3)
        self.assertEqual(mcm_obj2.dtype, tf.int32)

    def initialize_vars(self, n_classes, input_dtype):
        mcm_obj = MultiLabelConfusionMatrix(
            num_classes=n_classes, dtype=input_dtype)
        self.evaluate(tf.compat.v1.variables_initializer(mcm_obj.variables))
        return mcm_obj

    def update_obj_states(self, obj, actuals, preds):
        update_op = obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_mcm_3_classes(self):
        for input_dtype in [tf.int32, tf.int64, tf.float32, tf.float64]:
            actuals = tf.constant([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
                                  dtype=input_dtype)
            preds = tf.constant([[1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 1, 1]],
                                dtype=input_dtype)
            # Initialize
            mcm_obj = self.initialize_vars(
                n_classes=3, input_dtype=input_dtype)
            # Update
            self.update_obj_states(mcm_obj, actuals, preds)
            # Check results
            self.check_results(
                mcm_obj,
                [[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[0, 2], [2, 0]]])

    def test_mcm_4_classes(self):
        for input_dtype in [tf.int32, tf.int64, tf.float32, tf.float64]:
            actuals = tf.constant(
                [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1], [1, 1, 0, 0],
                 [0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
                 [0, 1, 1, 0], [0, 1, 0, 1]],
                dtype=input_dtype)
            preds = tf.constant(
                [[1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 0, 0],
                 [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1],
                 [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=input_dtype)

            # Initialize
            mcm_obj = self.initialize_vars(
                n_classes=4, input_dtype=input_dtype)
            # Update
            self.update_obj_states(mcm_obj, actuals, preds)
            # Check results
            self.check_results(mcm_obj, [[[4, 1], [1, 4]], [[6, 0], [2, 2]],
                                         [[6, 1], [1, 2]], [[2, 0], [2, 6]]])

    def test_multiclass(self):
        for input_dtype in [tf.int32, tf.int64, tf.float32, tf.float64]:
            actuals = tf.constant(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                 [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                 [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=input_dtype)
            preds = tf.constant(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0],
                 [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0],
                 [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=input_dtype)

            # Initialize
            mcm_obj = self.initialize_vars(
                n_classes=4, input_dtype=input_dtype)
            # Update
            self.update_obj_states(mcm_obj, actuals, preds)
            # Check results
            self.check_results(mcm_obj, [[[5, 2], [0, 3]], [[7, 1], [2, 0]],
                                         [[7, 0], [1, 2]], [[8, 0], [0, 2]]])
