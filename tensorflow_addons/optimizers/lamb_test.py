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
"""Tests for LAMB Optimizer."""

import numpy as np
from numpy import linalg

import tensorflow as tf

from tensorflow_addons.optimizers import lamb
from tensorflow_addons.utils import test_utils


def lamb_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      lr=0.001,
                      lamb_wd=0.0,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-6):

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    m_t_hat = m_t / (1 - beta1**(t + 1))
    v_t_hat = v_t / (1 - beta2**(t + 1))
    update = m_t_hat / (np.sqrt(v_t_hat) + epsilon)

    update += lamb_wd * param

    w_norm = linalg.norm(param, ord=2)
    g_norm = linalg.norm(update, ord=2)
    ratio = np.where(w_norm > 0, np.where(g_norm > 0, (w_norm / g_norm), 1.0),
                     1.0)

    param_t = param - ratio * lr * update
    return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
    local_step = tf.cast(opt.iterations + 1, dtype)
    beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


@test_utils.run_all_in_graph_and_eager_modes
class LAMBTest(tf.test.TestCase):
    # Based on issue #347 (https://github.com/tensorflow/addons/issues/347)
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel for 'GPU'.
    # So we have to remove tf.half when testing with gpu.
    def _DtypesToTest(self, use_gpu):
        if use_gpu:
            return [tf.float32, tf.float64]
        else:
            return [tf.half, tf.float32, tf.float64]

    def testSparse(self):
        for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
            # Initialize tf for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = tf.Variable(var0_np)
            var1 = tf.Variable(var1_np)
            grads0_np_indices = np.array([0, 2], dtype=np.int32)
            grads0 = tf.IndexedSlices(
                tf.constant(grads0_np[grads0_np_indices]),
                tf.constant(grads0_np_indices), tf.constant([3]))
            grads1_np_indices = np.array([0, 2], dtype=np.int32)
            grads1 = tf.IndexedSlices(
                tf.constant(grads1_np[grads1_np_indices]),
                tf.constant(grads1_np_indices), tf.constant([3]))
            opt = lamb.LAMB()
            if not tf.executing_eagerly():
                update = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            # Fetch params to validate initial values
            self.assertAllClose([1.0, 1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([3.0, 3.0, 4.0], self.evaluate(var1))

            # Run 3 steps of LAMB
            for t in range(3):
                beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
                self.assertAllCloseAccordingToType(0.9**(t + 1),
                                                   self.evaluate(beta_1_power))
                self.assertAllCloseAccordingToType(0.999**(t + 1),
                                                   self.evaluate(beta_2_power))
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0,
                                                    v0)
                var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1,
                                                    v1)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np,
                                                   self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np,
                                                   self.evaluate(var1))

    def doTestBasic(self, use_callable_params=False):
        for i, dtype in enumerate(
                self._DtypesToTest(use_gpu=tf.test.is_gpu_available())):
            # Initialize variables for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = tf.Variable(var0_np, name="var0_%d" % i)
            var1 = tf.Variable(var1_np, name="var1_%d" % i)
            grads0 = tf.constant(grads0_np)
            grads1 = tf.constant(grads1_np)

            learning_rate = lambda: 0.001
            beta1 = lambda: 0.9
            beta2 = lambda: 0.999
            epsilon = lambda: 1e-8
            if not use_callable_params:
                learning_rate = learning_rate()
                beta1 = beta1()
                beta2 = beta2()
                epsilon = epsilon()

            opt = lamb.LAMB(learning_rate=learning_rate)
            if not tf.executing_eagerly():
                update = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            # Run 3 steps of LAMB
            for t in range(3):
                beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
                self.assertAllCloseAccordingToType(0.9**(t + 1),
                                                   self.evaluate(beta_1_power))
                self.assertAllCloseAccordingToType(0.999**(t + 1),
                                                   self.evaluate(beta_2_power))
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0,
                                                    v0)
                var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1,
                                                    v1)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np,
                                                   self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np,
                                                   self.evaluate(var1))

    def testResourceBasic(self):
        self.doTestBasic()

    def testBasicCallableParams(self):
        self.doTestBasic(use_callable_params=True)

    def testBasicWithLearningRateDecay(self):
        for i, dtype in enumerate(
                self._DtypesToTest(use_gpu=tf.test.is_gpu_available())):
            # Initialize variables for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = tf.Variable(var0_np, name="var0_%d" % i)
            var1 = tf.Variable(var1_np, name="var1_%d" % i)
            grads0 = tf.constant(grads0_np)
            grads1 = tf.constant(grads1_np)

            learning_rate = 0.001
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 1e-7
            decay = 0.5
            lamb_wd = 0.01

            opt = lamb.LAMB(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                weight_decay_rate=lamb_wd,
                decay=decay)

            if not tf.executing_eagerly():
                update = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            # Run 3 steps of LAMB
            for t in range(3):
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                lr_np = learning_rate / (1 + decay * t)

                var0_np, m0, v0 = lamb_update_numpy(
                    var0_np, grads0_np, t, m0, v0, lr=lr_np, lamb_wd=lamb_wd)
                var1_np, m1, v1 = lamb_update_numpy(
                    var1_np, grads1_np, t, m1, v1, lr=lr_np, lamb_wd=lamb_wd)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np,
                                                   self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np,
                                                   self.evaluate(var1))

    def testBasicWithLearningRateInverseTimeDecay(self):
        for i, dtype in enumerate(
                self._DtypesToTest(use_gpu=tf.test.is_gpu_available())):
            # Initialize variables for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = tf.Variable(var0_np, name="var0_%d" % i)
            var1 = tf.Variable(var1_np, name="var1_%d" % i)
            grads0 = tf.constant(grads0_np)
            grads1 = tf.constant(grads1_np)

            learning_rate = 0.001
            decay = 0.5
            lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                learning_rate, decay_steps=1.0, decay_rate=decay)
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 1e-7

            opt = lamb.LAMB(
                learning_rate=lr_schedule,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon)

            if not tf.executing_eagerly():
                update = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            # Run 3 steps of LAMB
            for t in range(3):
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                lr_np = learning_rate / (1 + decay * t)

                var0_np, m0, v0 = lamb_update_numpy(
                    var0_np, grads0_np, t, m0, v0, lr=lr_np)
                var1_np, m1, v1 = lamb_update_numpy(
                    var1_np, grads1_np, t, m1, v1, lr=lr_np)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np,
                                                   self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np,
                                                   self.evaluate(var1))

    def testTensorLearningRate(self):
        for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
            # Initialize variables for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = tf.Variable(var0_np)
            var1 = tf.Variable(var1_np)
            grads0 = tf.constant(grads0_np)
            grads1 = tf.constant(grads1_np)
            opt = lamb.LAMB(tf.constant(0.001))

            if not tf.executing_eagerly():
                update = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            # Fetch params to validate initial values
            self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([3.0, 4.0], self.evaluate(var1))

            # Run 3 steps of LAMB
            for t in range(3):
                beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
                self.assertAllCloseAccordingToType(0.9**(t + 1),
                                                   self.evaluate(beta_1_power))
                self.assertAllCloseAccordingToType(0.999**(t + 1),
                                                   self.evaluate(beta_2_power))
                if not tf.executing_eagerly():
                    self.evaluate(update)
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0,
                                                    v0)
                var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1,
                                                    v1)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np,
                                                   self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np,
                                                   self.evaluate(var1))

    def testSharing(self):
        for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
            # Initialize variables for numpy implementation.
            m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
            var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
            grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
            var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
            grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

            var0 = tf.Variable(var0_np)
            var1 = tf.Variable(var1_np)
            grads0 = tf.constant(grads0_np)
            grads1 = tf.constant(grads1_np)
            opt = lamb.LAMB()

            if not tf.executing_eagerly():
                update1 = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                update2 = opt.apply_gradients(
                    zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            # Fetch params to validate initial values
            self.assertAllClose([1.0, 2.0], self.evaluate(var0))
            self.assertAllClose([3.0, 4.0], self.evaluate(var1))

            # Run 3 steps of intertwined LAMB1 and LAMB2.
            for t in range(3):
                beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
                self.assertAllCloseAccordingToType(0.9**(t + 1),
                                                   self.evaluate(beta_1_power))
                self.assertAllCloseAccordingToType(0.999**(t + 1),
                                                   self.evaluate(beta_2_power))

                if not tf.executing_eagerly():
                    if t % 2 == 0:
                        update1.run()
                    else:
                        update2.run()
                else:
                    opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0,
                                                    v0)
                var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1,
                                                    v1)

                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np,
                                                   self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np,
                                                   self.evaluate(var1))

    def testMinimizeMeanSquareLossWithWeightDecay(self):
        w = tf.Variable([0.1, -0.2, -0.1])
        x = tf.constant([0.4, 0.2, -0.5])
        loss = lambda: tf.reduce_mean(tf.square(x - w))  # pylint:disable=cell-var-from-loop
        opt = lamb.LAMB(0.02, weight_decay_rate=0.01)

        if not tf.executing_eagerly():
            op = opt.minimize(loss, [w])
            self.evaluate(tf.compat.v1.global_variables_initializer())

        self.evaluate(tf.compat.v1.global_variables_initializer())
        # Run 200 steps
        for _ in range(200):
            if tf.executing_eagerly():
                opt.minimize(loss, [w])
            else:
                self.evaluate(op)
        # Validate updated params
        self.assertAllClose(
            self.evaluate(w), [0.4, 0.2, -0.5], rtol=1e-2, atol=1e-2)

    def test_get_config(self):
        opt = lamb.LAMB(1e-4)
        config = opt.get_config()
        self.assertEqual(config['learning_rate'], 1e-4)


if __name__ == "__main__":
    tf.test.main()
