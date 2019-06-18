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
from tensorflow_addons.metrics import F1Score
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class F1ScoreTest(tf.test.TestCase):
    def test_config(self):
        f1_obj = F1Score(name='f1_score', num_classes=3)
        self.assertEqual(f1_obj.name, 'f1_score')
        self.assertEqual(f1_obj.dtype, tf.float32)
        self.assertEqual(f1_obj.num_classes, 3)
        # Check save and restore config
        f1_obj2 = F1Score.from_config(f1_obj.get_config())
        self.assertEqual(f1_obj2.name, 'f1_score')
        self.assertEqual(f1_obj2.dtype, tf.float32)
        self.assertEqual(f1_obj2.num_classes, 3)

    def initialize_vars(self):
        f1_micro = F1Score(num_classes=3, average='micro')
        f1_macro = F1Score(num_classes=3, average='macro')
        f1_weighted = F1Score(num_classes=3, average='weighted')

        self.evaluate(tf.compat.v1.variables_initializer(f1_micro.variables))
        self.evaluate(tf.compat.v1.variables_initializer(f1_macro.variables))
        self.evaluate(
            tf.compat.v1.variables_initializer(f1_weighted.variables))
        return f1_micro, f1_macro, f1_weighted

    def initialize_vars_none(self):
        f1_none = F1Score(num_classes=3, average=None)

        self.evaluate(tf.compat.v1.variables_initializer(f1_none.variables))
        return f1_none

    def update_obj_states(self, f1_micro, f1_macro, f1_weighted, actuals,
                          preds):
        update_micro = f1_micro.update_state(actuals, preds)
        update_macro = f1_macro.update_state(actuals, preds)
        update_weighted = f1_weighted.update_state(actuals, preds)
        self.evaluate(update_micro)
        self.evaluate(update_macro)
        self.evaluate(update_weighted)

    def update_obj_states_none(self, f1_none, actuals, preds):
        update_none = f1_none.update_state(actuals, preds)
        self.evaluate(update_none)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_f1_perfect_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=tf.int32)
        # Initialize
        f1_micro, f1_macro, f1_weighted = self.initialize_vars()
        # Update
        self.update_obj_states(f1_micro, f1_macro, f1_weighted, actuals, preds)
        # Check results
        self.check_results(f1_micro, 1.0)
        self.check_results(f1_macro, 1.0)
        self.check_results(f1_weighted, 1.0)

    def test_f1_worst_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int32)
        # Initialize
        f1_micro, f1_macro, f1_weighted = self.initialize_vars()
        # Update
        self.update_obj_states(f1_micro, f1_macro, f1_weighted, actuals, preds)
        # Check results
        self.check_results(f1_micro, 0.0)
        self.check_results(f1_macro, 0.0)
        self.check_results(f1_weighted, 0.0)

    def test_f1_random_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.int32)
        # Initialize
        f1_micro, f1_macro, f1_weighted = self.initialize_vars()
        # Update
        self.update_obj_states(f1_micro, f1_macro, f1_weighted, actuals, preds)
        # Check results
        self.check_results(f1_micro, 0.6666666)
        self.check_results(f1_macro, 0.6555555)
        self.check_results(f1_weighted, 0.6777777)

    def test_f1_none_score(self):
        actuals = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=tf.int32)
        preds = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
            dtype=tf.int32)
        # Initialize
        f1_none = self.initialize_vars_none()
        # Update
        self.update_obj_states_none(f1_none, actuals, preds)
        # Check results
        self.check_results(f1_none, [0.8, 0.6666667, 1.])


if __name__ == '__main__':
    tf.test.main()
