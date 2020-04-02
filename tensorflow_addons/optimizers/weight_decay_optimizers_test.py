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
"""Tests for optimizers with weight decay."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import weight_decay_optimizers

WEIGHT_DECAY = 0.01


class OptimizerTestBase(tf.test.TestCase):
    """Base class for optimizer tests.

    Optimizer tests may inherit from this class and define test
    functions using doTest. Usually this should include the functions
    testSparse, testBasic, and testBasicCallableParams. See
    weight_decay_optimizers_test for an example.
    """

    def doTest(
        self,
        optimizer,
        update_fn,
        do_sparse=False,
        do_decay_var_list=False,
        **optimizer_kwargs
    ):
        """The major test function.

        Args:
            optimizer: The tensorflow optimizer class to be tested.
            update_fn: The numpy update function of the optimizer, the function
                signature must be
                update_fn(var: np.array,
                          grad_t: np.array,
                          slot_vars: dict,
                          **kwargs) -> (updated_var, updated_slot_vars)
                Note that slot_vars will be initialized to an empty dictionary
                for each variable, initial values should be handled in the
                update_fn.
            do_sparse: If True, test sparse update. Defaults to False, i.e.,
                dense update.
            do_decay_var_list: If True, test by passing a list of vars to ensure hashing is handled correctly
            **optimizer_kwargs:The parameters to pass to the construcor of the
                optimizer. Either a constant or a callable. This also passed to
                the optimizer_params in the update_fn.
        """
        # TODO: Fix #347 issue
        if do_sparse and tf.test.is_gpu_available():
            self.skipTest("Wait #347 to be fixed")

        for i, dtype in enumerate([tf.half, tf.float32, tf.float64]):
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
                    tf.constant(grads0_np),
                    tf.constant(grads0_np_indices),
                    tf.constant([2]),
                )
                grads1_np_indices = np.array([0, 1], dtype=np.int32)
                grads1 = tf.IndexedSlices(
                    tf.constant(grads1_np),
                    tf.constant(grads1_np_indices),
                    tf.constant([2]),
                )
            else:
                grads0 = tf.constant(grads0_np)
                grads1 = tf.constant(grads1_np)
            opt = optimizer(**optimizer_kwargs)
            # Validate initial values.
            if not tf.executing_eagerly():
                if do_decay_var_list:
                    update = opt.apply_gradients(
                        zip([grads0, grads1], [var0, var1]), decay_var_list=[var0, var1]
                    )
                else:
                    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                self.evaluate(tf.compat.v1.global_variables_initializer())
                self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                self.assertAllClose([3.0, 4.0], self.evaluate(var1))
            # Create the update op.
            # Run 3 steps of the optimizer
            for _ in range(3):
                if tf.executing_eagerly():
                    if do_decay_var_list:
                        opt.apply_gradients(
                            zip([grads0, grads1], [var0, var1]),
                            decay_var_list=[var0, var1],
                        )
                    else:
                        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                else:
                    self.evaluate(update)
                var0_np, np_slot_vars0 = update_fn(
                    var0_np, grads0_np, np_slot_vars0, **optimizer_kwargs
                )
                var1_np, np_slot_vars1 = update_fn(
                    var1_np, grads1_np, np_slot_vars1, **optimizer_kwargs
                )
                # Validate updated params
                self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
                self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

    def doTestSparseRepeatedIndices(self, optimizer, **optimizer_kwargs):
        """Test for repeated indices in sparse updates.

        This test verifies that an update with repeated indices is the same as
        an update with two times the gradient.

        Args:
            optimizer: The tensorflow optimizer class to be tested.
            **optimizer_kwargs: The parameters to pass to the construcor of the
                optimizer. Either a constant or a callable. This also passed to
                the optimizer_params in the update_fn.
        """
        # TODO: Fix #347 issue
        if tf.test.is_gpu_available():
            self.skipTest("Wait #347 to be fixed")

        for dtype in [tf.dtypes.half, tf.dtypes.float32, tf.dtypes.float64]:
            repeated_index_update_var = tf.Variable([[1.0], [2.0]], dtype=dtype)
            aggregated_update_var = tf.Variable([[1.0], [2.0]], dtype=dtype)
            grad_repeated_index = tf.IndexedSlices(
                tf.constant([0.1, 0.1], shape=[2, 1], dtype=dtype),
                tf.constant([1, 1]),
                tf.constant([2, 1]),
            )
            grad_aggregated = tf.IndexedSlices(
                tf.constant([0.2], shape=[1, 1], dtype=dtype),
                tf.constant([1]),
                tf.constant([2, 1]),
            )
            opt_repeated = optimizer(**optimizer_kwargs)
            repeated_update = opt_repeated.apply_gradients(
                [(grad_repeated_index, repeated_index_update_var)]
            )
            opt_aggregated = optimizer(**optimizer_kwargs)
            aggregated_update = opt_aggregated.apply_gradients(
                [(grad_aggregated, aggregated_update_var)]
            )
            self.evaluate(tf.compat.v1.global_variables_initializer())
            self.assertAllClose(
                self.evaluate(aggregated_update_var),
                self.evaluate(repeated_index_update_var),
            )
            for _ in range(3):
                if not tf.executing_eagerly():
                    self.evaluate(repeated_update)
                    self.evaluate(aggregated_update)
                else:
                    opt_repeated.apply_gradients(
                        [(grad_repeated_index, repeated_index_update_var)]
                    )
                    opt_aggregated.apply_gradients(
                        [(grad_aggregated, aggregated_update_var)]
                    )
                self.assertAllClose(
                    self.evaluate(aggregated_update_var),
                    self.evaluate(repeated_index_update_var),
                )


def adamw_update_numpy(
    param, grad_t, slot_vars, learning_rate, beta_1, beta_2, epsilon, weight_decay
):
    """Numpy update function for AdamW."""
    lr, beta1, beta2, eps, wd = (
        v() if callable(v) else v
        for v in (learning_rate, beta_1, beta_2, epsilon, weight_decay)
    )
    t = slot_vars.get("t", 0) + 1
    lr_t = lr * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    slot_vars["m"] = beta1 * slot_vars.get("m", 0) + (1 - beta1) * grad_t
    slot_vars["v"] = beta2 * slot_vars.get("v", 0) + (1 - beta2) * grad_t ** 2
    param_t = param * (1 - wd) - lr_t * slot_vars["m"] / (np.sqrt(slot_vars["v"]) + eps)
    slot_vars["t"] = t
    return param_t, slot_vars


def sgdw_update_numpy(param, grad_t, slot_vars, learning_rate, momentum, weight_decay):
    """Numpy update function for SGDW."""
    m = slot_vars.get("m", 0)
    lr, momentum, wd = (
        v() if callable(v) else v for v in (learning_rate, momentum, weight_decay)
    )
    slot_vars["m"] = momentum * m + grad_t
    param_t = param * (1 - wd) - lr * slot_vars["m"]
    return param_t, slot_vars


@test_utils.run_all_in_graph_and_eager_modes
class AdamWTest(OptimizerTestBase):

    optimizer = weight_decay_optimizers.AdamW

    def testSparse(self):
        self.doTest(
            self.optimizer,
            adamw_update_numpy,
            do_sparse=True,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=WEIGHT_DECAY,
        )

    def testSparseRepeatedIndices(self):
        self.doTestSparseRepeatedIndices(
            self.optimizer,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=WEIGHT_DECAY,
        )

    def testBasic(self):
        self.doTest(
            self.optimizer,
            adamw_update_numpy,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=WEIGHT_DECAY,
        )

    def testBasicCallableParams(self):
        self.doTest(
            self.optimizer,
            adamw_update_numpy,
            learning_rate=lambda: 0.001,
            beta_1=lambda: 0.9,
            beta_2=lambda: 0.999,
            epsilon=1e-8,
            weight_decay=lambda: WEIGHT_DECAY,
        )

    def testBasicDecayVarList(self):
        self.doTest(
            self.optimizer,
            adamw_update_numpy,
            do_decay_var_list=True,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=WEIGHT_DECAY,
        )


@test_utils.run_all_in_graph_and_eager_modes
class SGDWTest(OptimizerTestBase):

    optimizer = weight_decay_optimizers.SGDW

    def testSparse(self):
        self.doTest(
            self.optimizer,
            sgdw_update_numpy,
            do_sparse=True,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    def testSparseRepeatedIndices(self):
        self.doTestSparseRepeatedIndices(
            self.optimizer, learning_rate=0.001, momentum=0.9, weight_decay=WEIGHT_DECAY
        )

    def testBasic(self):
        self.doTest(
            self.optimizer,
            sgdw_update_numpy,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )

    def testBasicCallableParams(self):
        self.doTest(
            self.optimizer,
            sgdw_update_numpy,
            learning_rate=lambda: 0.001,
            momentum=lambda: 0.9,
            weight_decay=lambda: WEIGHT_DECAY,
        )

    def testBasicDecayVarList(self):
        self.doTest(
            self.optimizer,
            sgdw_update_numpy,
            do_decay_var_list=True,
            learning_rate=0.001,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY,
        )


class ExtendWithWeightDecayTest(SGDWTest):
    """Verify that the factory function SGDW is the same as SGDW."""

    optimizer = weight_decay_optimizers.extend_with_decoupled_weight_decay(
        tf.keras.optimizers.SGD
    )
