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
"""Tests for R-Square Metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import RSquare
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class RSquareTest(tf.test.TestCase):
    def test_config(self):
        r2_obj = RSquare(name='r_square')
        self.assertEqual(r2_obj.name, 'r_square')
        self.assertEqual(r2_obj.dtype, tf.float32)
        # Check save and restore config
        r2_obj2 = RSquare.from_config(r2_obj.get_config())
        self.assertEqual(r2_obj2.name, 'r_square')
        self.assertEqual(r2_obj2.dtype, tf.float32)

    def initialize_vars(self):
        r2_obj = RSquare()
        self.evaluate(tf.compat.v1.variables_initializer(r2_obj.variables))
        return r2_obj

    def update_obj_states(self, obj, actuals, preds):
        update_op = obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_r2_perfect_score(self):
        actuals = tf.constant([100, 700, 40, 5.7], dtype=tf.float32)
        preds = tf.constant([100, 700, 40, 5.7], dtype=tf.float32)
        actuals = tf.cast(actuals, dtype=tf.float32)
        preds = tf.cast(preds, dtype=tf.float32)
        # Initialize
        r2_obj = self.initialize_vars()
        # Update
        self.update_obj_states(r2_obj, actuals, preds)
        # Check results
        self.check_results(r2_obj, 1.0)

    def test_r2_worst_score(self):
        actuals = tf.constant([10, 600, 4, 9.77], dtype=tf.float32)
        preds = tf.constant([1, 70, 40, 5.7], dtype=tf.float32)
        actuals = tf.cast(actuals, dtype=tf.float32)
        preds = tf.cast(preds, dtype=tf.float32)
        # Initialize
        r2_obj = self.initialize_vars()
        # Update
        self.update_obj_states(r2_obj, actuals, preds)
        # Check results
        self.check_results(r2_obj, -0.073607)

    def test_r2_random_score(self):
        actuals = tf.constant([10, 600, 3, 9.77], dtype=tf.float32)
        preds = tf.constant([1, 340, 40, 5.7], dtype=tf.float32)
        actuals = tf.cast(actuals, dtype=tf.float32)
        preds = tf.cast(preds, dtype=tf.float32)
        # Initialize
        r2_obj = self.initialize_vars()
        # Update
        self.update_obj_states(r2_obj, actuals, preds)
        # Check results
        self.check_results(r2_obj, 0.7376327)


if __name__ == '__main__':
    tf.test.main()
