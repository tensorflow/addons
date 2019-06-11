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
"""Tests for Cohen's Kappa Metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import CohenKappa
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class CohenKappaTest(tf.test.TestCase):
    def test_config(self):
        kp_obj = CohenKappa(name='cohen_kappa', num_classes=5)
        self.assertEqual(kp_obj.name, 'cohen_kappa')
        self.assertEqual(kp_obj.dtype, tf.float32)
        self.assertEqual(kp_obj.num_classes, 5)

        # Check save and restore config
        kb_obj2 = CohenKappa.from_config(kp_obj.get_config())
        self.assertEqual(kb_obj2.name, 'cohen_kappa')
        self.assertEqual(kb_obj2.dtype, tf.float32)
        self.assertEqual(kp_obj.num_classes, 5)

    def initialize_vars(self):
        kp_obj1 = CohenKappa(num_classes=5)
        kp_obj2 = CohenKappa(num_classes=5, weightage='linear')
        kp_obj3 = CohenKappa(num_classes=5, weightage='quadratic')

        self.evaluate(tf.compat.v1.variables_initializer(kp_obj1.variables))
        self.evaluate(tf.compat.v1.variables_initializer(kp_obj2.variables))
        self.evaluate(tf.compat.v1.variables_initializer(kp_obj3.variables))

        return kp_obj1, kp_obj2, kp_obj3

    def update_obj_states(self, obj1, obj2, obj3, actuals, preds, weights):
        update_op1 = obj1.update_state(actuals, preds, sample_weight=weights)
        update_op2 = obj2.update_state(actuals, preds, sample_weight=weights)
        update_op3 = obj3.update_state(actuals, preds, sample_weight=weights)

        self.evaluate(update_op1)
        self.evaluate(update_op2)
        self.evaluate(update_op3)

    def test_kappa_random_score(self):
        actuals = [4, 4, 3, 4, 2, 4, 1, 1]
        preds = [4, 4, 3, 4, 4, 2, 1, 1]
        actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
        preds = tf.convert_to_tensor(preds, dtype=tf.int32)

        # Initialize
        kp_obj1, kp_obj2, kp_obj3 = self.initialize_vars()

        # Update
        self.update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, None)

        # Check results
        self.assertAlmostEqual(self.evaluate(kp_obj1.result()), 0.61904, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj2.result()), 0.62790, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj3.result()), 0.68932, 4)

    def test_kappa_perfect_score(self):
        actuals = [4, 4, 3, 3, 2, 2, 1, 1]
        preds = [4, 4, 3, 3, 2, 2, 1, 1]
        actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
        preds = tf.convert_to_tensor(preds, dtype=tf.int32)

        # Initialize
        kp_obj1, kp_obj2, kp_obj3 = self.initialize_vars()

        # Update
        self.update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, None)

        self.assertAlmostEqual(self.evaluate(kp_obj1.result()), 1.0, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj2.result()), 1.0, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj3.result()), 1.0, 4)

    def test_kappa_worse_than_random(self):
        actuals = [4, 4, 3, 3, 2, 2, 1, 1]
        preds = [1, 2, 4, 1, 3, 3, 4, 4]
        actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
        preds = tf.convert_to_tensor(preds, dtype=tf.int32)

        # Initialize
        kp_obj1, kp_obj2, kp_obj3 = self.initialize_vars()

        # Update
        self.update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, None)

        self.assertAlmostEqual(self.evaluate(kp_obj1.result()), -0.33333, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj2.result()), -0.52380, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj3.result()), -0.72727, 4)

    def test_kappa_with_sample_weights(self):
        actuals = [4, 4, 3, 3, 2, 2, 1, 1]
        preds = [1, 2, 4, 1, 3, 3, 4, 4]
        weights = [1, 1, 2, 5, 10, 2, 3, 3]
        actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
        preds = tf.convert_to_tensor(preds, dtype=tf.int32)
        weights = tf.convert_to_tensor(weights, dtype=tf.int32)

        # Initialize
        kp_obj1, kp_obj2, kp_obj3 = self.initialize_vars()

        # Update
        self.update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds,
                               weights)

        self.assertAlmostEqual(self.evaluate(kp_obj1.result()), -0.254733, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj2.result()), -0.389923, 4)
        self.assertAlmostEqual(self.evaluate(kp_obj3.result()), -0.606953, 4)
