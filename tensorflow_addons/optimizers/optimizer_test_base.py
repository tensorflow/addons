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
"""Base class for optimizer tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class OptimizerTestBase(tf.test.TestCase):
    """Base class for optimizer tests.

    Optimizer tests may inherit from this class and define test
    functions using doTest. Usually this should include the functions
    testSparse, testBasic, and testBasicCallableParams. See
    weight_decay_optimizers_test for an example.
    """

    def doTest(self, optimizer, update_fn, params, do_sparse=False):
        """The major test function.

        Args:
            optimizer: The tensorflow optimizer class to be tested.
            update_fn: The numpy update function of the optimizer, the function
                signature must be
                update_fn(var: np.array,
                          grad_t: np.array,
                          slot_vars: dict,
                          optimizer_params: dict) -> (updated_var,
                                                      updated_slot_vars)
                Note that slot_vars will be initialized to an empty dictionary
                for each variable, initial values should be handled in the
                update_fn.
            params: A dict, the parameters to pass to the construcor of the
                optimizer. Either a constant or a callable. This also passed to
                    the optimizer_params in the update_fn.
            do_sparse: If True, test sparse update. Defaults to False, i.e.,
                dense update.
        """
        for i, dtype in enumerate([tf.half, tf.float32, tf.float64]):
            with self.session(graph=tf.Graph()):
                # Initialize variables for numpy implementation.
                np_slot_vars0, np_slot_vars1 = {}, {}
                var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
                grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
                var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
                grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)
                # Create Tensorflow variables.
                var0 = tf.Variable(var0_np, name="var0_%d" % i)
                var1 = tf.Variable(var1_np, name="var1_%d" % i)
                if do_sparse:
                    grads0_np_indices = np.array([0, 1], dtype=np.int32)
                    grads0 = tf.IndexedSlices(
                        tf.constant(grads0_np), tf.constant(grads0_np_indices),
                        tf.constant([2]))
                    grads1_np_indices = np.array([0, 1], dtype=np.int32)
                    grads1 = tf.IndexedSlices(
                        tf.constant(grads1_np), tf.constant(grads1_np_indices),
                        tf.constant([2]))
                else:
                    grads0 = tf.constant(grads0_np)
                    grads1 = tf.constant(grads1_np)
                opt = optimizer(**params)
                # Validate initial values.
                if not tf.executing_eagerly():
                    update = opt.apply_gradients(
                        zip([grads0, grads1], [var0, var1]))
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                    self.assertAllClose([3.0, 4.0], self.evaluate(var1))
                # Create the update op.
                # Run 3 steps of the optimizer
                for _ in range(3):
                    if tf.executing_eagerly():
                        opt.apply_gradients(
                            zip([grads0, grads1], [var0, var1]))
                    else:
                        self.evaluate(update)
                    var0_np, np_slot_vars0 = update_fn(var0_np, grads0_np,
                                                       np_slot_vars0, params)
                    var1_np, np_slot_vars1 = update_fn(var1_np, grads1_np,
                                                       np_slot_vars1, params)
                    # Validate updated params
                    self.assertAllCloseAccordingToType(var0_np,
                                                       self.evaluate(var0))
                    self.assertAllCloseAccordingToType(var1_np,
                                                       self.evaluate(var1))

    def doTestSparseRepeatedIndices(self, optimizer, params):
        """Test for repeated indices in sparse updates.

        This test verifies that an update with repeated indices is the same as
        an update with two times the gradient.

        Args:
            optimizer: The tensorflow optimizer class to be tested.
            params: A dict, the parameters to pass to the construcor of the
                optimizer. Either a constant or a callable. This also passed to
                the optimizer_params in the update_fn.
        """
        for dtype in [tf.dtypes.half, tf.dtypes.float32, tf.dtypes.float64]:
            with self.cached_session():
                repeated_index_update_var = tf.Variable([[1.0], [2.0]],
                                                        dtype=dtype)
                aggregated_update_var = tf.Variable([[1.0], [2.0]],
                                                    dtype=dtype)
                grad_repeated_index = tf.IndexedSlices(
                    tf.constant([0.1, 0.1], shape=[2, 1], dtype=dtype),
                    tf.constant([1, 1]), tf.constant([2, 1]))
                grad_aggregated = tf.IndexedSlices(
                    tf.constant([0.2], shape=[1, 1], dtype=dtype),
                    tf.constant([1]), tf.constant([2, 1]))
                opt_repeated = optimizer(**params)
                repeated_update = opt_repeated.apply_gradients(
                    [(grad_repeated_index, repeated_index_update_var)])
                opt_aggregated = optimizer(**params)
                aggregated_update = opt_aggregated.apply_gradients(
                    [(grad_aggregated, aggregated_update_var)])
                self.evaluate(tf.compat.v1.global_variables_initializer())
                self.assertAllClose(
                    self.evaluate(aggregated_update_var),
                    self.evaluate(repeated_index_update_var))
                for _ in range(3):
                    if not tf.executing_eagerly():
                        self.evaluate(repeated_update)
                        self.evaluate(aggregated_update)
                    else:
                        opt_repeated.apply_gradients(
                            [(grad_repeated_index, repeated_index_update_var)])
                        opt_aggregated.apply_gradients(
                            [(grad_aggregated, aggregated_update_var)])
                    self.assertAllClose(
                        self.evaluate(aggregated_update_var),
                        self.evaluate(repeated_index_update_var))
