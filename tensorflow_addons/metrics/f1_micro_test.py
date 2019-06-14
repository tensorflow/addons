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
"""Tests F1 micro metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import F1Micro
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class F1MicroTest(tf.test.TestCase):
    def test_config(self):
        f1_micro_obj = F1Micro(name='f1_micro_score')
        self.assertEqual(f1_micro_obj.name, 'f1_micro_score')
        self.assertEqual(f1_micro_obj.dtype, tf.float32)
        # Check save and restore config
        f1_micro_obj2 = F1Micro.from_config(f1_micro_obj.get_config())
        self.assertEqual(f1_micro_obj2.name, 'f1_micro_score')
        self.assertEqual(f1_micro_obj2.dtype, tf.float32)

    def initialize_vars(self):
        f1_micro_obj = F1Micro()
        self.evaluate(tf.compat.v1.variables_initializer(
            f1_micro_obj.variables))
        return f1_micro_obj

    def update_obj_states(self, obj, actuals, preds):
        update_op = obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_f1_micro_perfect_score(self):
        actuals = [[1, 1, 0], [1, 0, 0], [1, 1, 0]]
        preds = [[1, 1, 0], [1, 0, 0], [1, 1, 0]]
        actuals = tf.constant(actuals, dtype=tf.float32)
        preds = tf.constant(preds, dtype=tf.float32)
        # Initialize
        f1_micro_obj = self.initialize_vars()
        # Update
        self.update_obj_states(f1_micro_obj, actuals, preds)
        # Check results
        self.check_results(f1_micro_obj, 1.0)

    def test_f1_micro_worst_score(self):
        actuals = [[1, 1, 0], [1, 0, 0], [1, 1, 0]]
        preds = [[0, 0, 0], [0, 1, 1], [0, 0, 0]]
        actuals = tf.constant(actuals, dtype=tf.float32)
        preds = tf.constant(preds, dtype=tf.float32)
        f1_micro_obj = self.initialize_vars()
        # Update
        self.update_obj_states(f1_micro_obj, actuals, preds)
        # Check results
        self.check_results(f1_micro_obj, 0.0)

    def test_f1_micro_random_score(self):
        actuals = [[1, 1, 0], [1, 0, 0], [1, 1, 0]]
        preds = [[1, 1, 0], [1, 1, 1], [0, 1, 0]]
        actuals = tf.constant(actuals, dtype=tf.float32)
        preds = tf.constant(preds, dtype=tf.float32)
        f1_micro_obj = self.initialize_vars()
        # Update
        self.update_obj_states(f1_micro_obj, actuals, preds)
        # Check results
        self.check_results(f1_micro_obj, 0.7272727)


if __name__ == '__main__':
    tf.test.main()
