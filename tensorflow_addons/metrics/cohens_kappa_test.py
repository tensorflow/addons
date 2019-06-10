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

import numpy as np
import tensorflow as tf

from tensorflow_addons.metrics import CohensKappa
from tensorflow_addons.utils import test_utils

@test_utils.run_all_in_graph_and_eager_modes
class CohensKappaTest(tf.test.TestCase):
  def test_config(self):
    kp_obj = CohensKappa(name='cohens_kappa')
    self.assertEqual(kp_obj.name, 'cohens_kappa')
  
  def test_kappa_random_score(self):
    kp_obj1 = CohensKappa(num_classes=5)
    kp_obj2 = CohensKappa(num_classes=5, weightage='linear')
    kp_obj3 = CohensKappa(num_classes=5, weightage='quadratic')
    
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj1.variables))
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj2.variables))
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj3.variables))
    
    # random score
    actuals = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
    preds = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)
    actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
    preds = tf.convert_to_tensor(preds, dtype=tf.int32)

    update_op1 = kp_obj1.update_state(actuals, preds)
    update_op2 = kp_obj2.update_state(actuals, preds)
    update_op3 = kp_obj3.update_state(actuals, preds)
    
    self.evaluate(update_op1)
    self.evaluate(update_op2)
    self.evaluate(update_op3)

    self.assertAlmostEqual(self.evaluate(kp_obj1.result()), 0.61904, 4)
    self.assertAlmostEqual(self.evaluate(kp_obj2.result()), 0.62790, 4)
    self.assertAlmostEqual(self.evaluate(kp_obj3.result()), 0.68932, 4)

  def test_kappa_perfect_score(self):
    kp_obj1 = CohensKappa(num_classes=5)
    kp_obj2 = CohensKappa(num_classes=5, weightage='linear')
    kp_obj3 = CohensKappa(num_classes=5, weightage='quadratic')
    
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj1.variables))
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj2.variables))
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj3.variables))
    
    # perfect score
    actuals = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
    preds = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
    actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
    preds = tf.convert_to_tensor(preds, dtype=tf.int32)

    update_op1 = kp_obj1.update_state(actuals, preds)
    update_op2 = kp_obj2.update_state(actuals, preds)
    update_op3 = kp_obj3.update_state(actuals, preds)
    
    self.evaluate(update_op1)
    self.evaluate(update_op2)
    self.evaluate(update_op3)
    
    
    self.assertAlmostEqual(self.evaluate(kp_obj1.result()), 1.0, 4)
    self.assertAlmostEqual(self.evaluate(kp_obj2.result()), 1.0, 4)
    self.assertAlmostEqual(self.evaluate(kp_obj3.result()), 1.0, 4)

  def test_kappa_worse_than_random(self):
    kp_obj1 = CohensKappa(num_classes=5)
    kp_obj2 = CohensKappa(num_classes=5, weightage='linear')
    kp_obj3 = CohensKappa(num_classes=5, weightage='quadratic')
    
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj1.variables))
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj2.variables))
    self.evaluate(tf.compat.v1.variables_initializer(kp_obj3.variables))
    
    # worse than random 
    actuals = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
    preds = np.array([1, 2, 4, 1, 3, 3, 4, 4], dtype=np.int32)
    actuals = tf.convert_to_tensor(actuals, dtype=tf.int32)
    preds = tf.convert_to_tensor(preds, dtype=tf.int32)

    update_op1 = kp_obj1.update_state(actuals, preds)
    update_op2 = kp_obj2.update_state(actuals, preds)
    update_op3 = kp_obj3.update_state(actuals, preds)
    
    self.evaluate(update_op1)
    self.evaluate(update_op2)
    self.evaluate(update_op3)

    self.assertAlmostEqual(self.evaluate(kp_obj1.result()), -0.33333, 4)
    self.assertAlmostEqual(self.evaluate(kp_obj2.result()), -0.52380, 4)
    self.assertAlmostEqual(self.evaluate(kp_obj3.result()), -0.72727, 4)

if __name__ == '__main__':
  tf.test.main() 