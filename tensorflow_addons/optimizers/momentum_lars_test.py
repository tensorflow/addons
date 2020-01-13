# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0. Licensed to the Apache
# Software Foundation. You may not use this file except in compliance with the
# License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for Layer-wise Adaptive Rate Scaling optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_addons.utils import test_utils
import numpy as np
from tensorflow_addons.optimizers import momentum_lars as lo
import tensorflow_addons.optimizers.conditional_gradient as cg_lib

@test_utils.run_all_in_graph_and_eager_modes
class MomentumLARSTest(tf.test.TestCase):

  def testLARSGradientOneStep(self, use_resource=False, use_callable_params=False):
      for dtype in [tf.float16, tf.float32, tf.float64]:
          shape = [3, 3]
          var_np = np.ones(shape)
          grad_np = np.full((3, 3), 3)
          lr_np = 0.1
          m_np = 0.9
          wd_np = 0.1
          ep_np = 1e-5
          eeta = 0.1
          vel_np = np.zeros(shape)

          var = tf.Variable(var_np, dtype=dtype)
          grad = tf.constant(grad_np, dtype=dtype)
          opt = lo.LARSOptimizer(
               learning_rate=lr_np,
               momentum=m_np,
               weight_decay=wd_np,
               eeta=eeta,
               epsilon=ep_np)

          update = opt.apply_gradients([(grad, var)])

          if not tf.executing_eagerly():
            self.evaluate(tf.compat.v1.global_variables_initializer())
            # Fetch params to validate initial values
            var_tmp = np.ones(shape)
            self.assertAllClose(var_tmp, self.evaluate(var))

          # Run 1 step of the optimizer.
          self.evaluate(update)
          post_var = self.evaluate(var)

          # Check we have slots
          self.assertEqual(["momentum"], opt.get_slot_names())
          slot0 = opt.get_slot(var, "momentum")
          self.assertEqual(slot0.get_shape(), var.get_shape())
          post_vel = self.evaluate(opt.get_slot(var, 'momentum'))

          w_norm = np.linalg.norm(var_np.flatten(), ord=2)
          g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
          trust_ratio = eeta * w_norm / (g_norm + wd_np * w_norm + ep_np)
          scaled_lr = lr_np * trust_ratio
          grad_np = grad_np + wd_np * var_np

          vel_np = m_np * vel_np + scaled_lr * grad_np
          var_np -= vel_np

          self.assertAllClose(var_np, post_var, 1e-3, 1e-3)
          self.assertAllClose(vel_np, post_vel, 1e-3, 1e-3)

  def testLARSGradientMultiStep(self, use_resource=False, use_callable_params=False):
      for dtype in [tf.float16, tf.float32, tf.float64]:
           shape = [3, 3]
           var_np = np.ones(shape)
           grad_np = np.ones(shape)
           lr_np = 0.1
           m_np = 0.9
           wd_np = 0.1
           ep_np = 1e-5
           eeta = 0.1
           vel_np = np.zeros(shape)
           iterations = 10

           var = tf.Variable(var_np, dtype=dtype)
           grad = tf.Variable(grad_np, dtype=dtype)
           opt = lo.LARSOptimizer(
               learning_rate=lr_np,
               momentum=m_np,
               eeta=eeta,
               weight_decay=wd_np,
               epsilon=ep_np)

           if not tf.executing_eagerly():
             self.evaluate(tf.compat.v1.global_variables_initializer())
             # Fetch params to validate initial values
             var_tmp = np.ones(shape)
             self.assertAllClose(var_tmp, self.evaluate(var))

           # initialize the variables for eager mode.
           if not tf.executing_eagerly():
             update = opt.apply_gradients([(grad, var)])
             self.evaluate(tf.compat.v1.global_variables_initializer())

           for _ in range(iterations):
             if tf.executing_eagerly():
               opt.apply_gradients([(grad, var)])
             else:
               self.evaluate(update)

             post_var = self.evaluate(var)

             # Check we have slots
             self.assertEqual(["momentum"], opt.get_slot_names())
             slot0 = opt.get_slot(var, "momentum")
             self.assertEqual(slot0.get_shape(), var.get_shape())
             post_vel = self.evaluate(opt.get_slot(var, 'momentum'))

             w_norm = np.linalg.norm(var_np.flatten(), ord=2)
             g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
             trust_ratio = eeta * w_norm / (g_norm + wd_np * w_norm + ep_np)
             scaled_lr = lr_np * trust_ratio
             grad_np = grad_np + wd_np * var_np

             vel_np = m_np * vel_np + scaled_lr * grad_np
             var_np -= vel_np

             self.assertAllClose(var_np, post_var, 1e-3, 1e-3)
             self.assertAllClose(vel_np, post_vel, 1e-3, 1e-3)

  def testGetConfig(self, use_resource=False, use_callable_params=False):
    lr_np = 0.1
    m_np = 0.9
    eeta = 0.1
    wd_np = 0.1
    ep_np = 1e-5
    opt = lo.LARSOptimizer(
        learning_rate=lr_np,
        momentum=m_np,
        eeta=eeta,
        weight_decay=wd_np,
        epsilon=ep_np)

    opt = tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(opt))
    config = opt.get_config()
    self.assertEqual(config['learning_rate'], 0.1)
    self.assertEqual(config['momentum'], 0.9)
    self.assertEqual(config['weight_decay'], 0.1)
    self.assertEqual(config['eeta'], 0.1)
    self.assertEqual(config['epsilon'], 1e-5)
    self.assertEqual(config['use_nesterov'], False)


if __name__ == '__main__':
  tf.test.main()
