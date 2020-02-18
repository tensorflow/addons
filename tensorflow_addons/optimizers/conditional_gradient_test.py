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
"""Tests for Conditional Gradient."""

import tensorflow as tf
from tensorflow_addons.utils import test_utils
import numpy as np
import conditional_gradient as cg_lib


@test_utils.run_all_in_graph_and_eager_modes
class ConditionalGradientTest(tf.test.TestCase):
    def _update_conditional_gradient_numpy(self, var, norm, g, lr, lambda_):
        var = var * lr - (1 - lr) * lambda_ * g / norm
        return var

    def top_singular_vector(self, m):
        # handle the case where m is a tensor of rank 0 or rank 1.
        n = tf.cond(
            tf.equal(tf.rank(m), 0),
            lambda: tf.expand_dims(tf.expand_dims(m, [0]), [0]),
            lambda: m,
        )
        n = tf.cond(tf.equal(tf.rank(m), 1), lambda: tf.expand_dims(m, [0]), lambda: n,)
        st, ut, vt = tf.linalg.svd(n, full_matrices=False)
        m_size = tf.shape(n)
        ut = tf.reshape(ut[:, 0], [m_size[0], 1])
        vt = tf.reshape(vt[:, 0], [m_size[1], 1])
        st = tf.matmul(ut, tf.transpose(vt))
        # when we return the top singular vector, we have to remove the
        # dimension we have added on
        st = tf.cond(
            tf.equal(tf.rank(m), 0),
            lambda: tf.squeeze(tf.squeeze(st, [0]), [0]),
            lambda: st,
        )
        st = tf.cond(tf.equal(tf.rank(m), 1), lambda: tf.squeeze(st, [0]), lambda: st,)
        return st

    def doTestBasicFrobenius(self, use_resource=False, use_callable_params=False):
        for i, dtype in enumerate([tf.half, tf.float32, tf.float64]):
            if use_resource:
                var0 = tf.Variable([1.0, 2.0], dtype=dtype, name="var0_%d" % i)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype, name="var1_%d" % i)
            else:
                var0 = tf.Variable([1.0, 2.0], dtype=dtype)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype)
            grads0 = tf.constant([0.1, 0.1], dtype=dtype)
            grads1 = tf.constant([0.01, 0.01], dtype=dtype)
            norm0 = tf.math.reduce_sum(grads0 ** 2) ** 0.5
            norm1 = tf.math.reduce_sum(grads1 ** 2) ** 0.5

            def learning_rate():
                return 0.5

            def lambda_():
                return 0.01

            ord = "fro"

            if not use_callable_params:
                learning_rate = learning_rate()
                lambda_ = lambda_()

            cg_opt = cg_lib.ConditionalGradient(
                learning_rate=learning_rate, lambda_=lambda_, ord=ord
            )
            cg_update = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            if not tf.executing_eagerly():
                self.evaluate(tf.compat.v1.global_variables_initializer())
                # Fetch params to validate initial values
                self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                self.assertAllClose([3.0, 4.0], self.evaluate(var1))

            # Check we have slots
            self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
            slot0 = cg_opt.get_slot(var0, "conditional_gradient")
            self.assertEquals(slot0.get_shape(), var0.get_shape())
            slot1 = cg_opt.get_slot(var1, "conditional_gradient")
            self.assertEquals(slot1.get_shape(), var1.get_shape())

            if not tf.executing_eagerly():
                self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                self.assertFalse(slot1 in tf.compat.v1.trainable_variables())

            if not tf.executing_eagerly():
                self.evaluate(cg_update)

            # Check that the parameters have been updated.
            norm0 = self.evaluate(norm0)
            norm1 = self.evaluate(norm1)

            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
                        2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
                    ]
                ),
                self.evaluate(var0),
            )
            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
                        4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
                    ]
                ),
                self.evaluate(var1),
            )

            # Step 2: the conditional_gradient contain the previous update.
            if tf.executing_eagerly():
                cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            else:
                self.evaluate(cg_update)
            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        (1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                        - (1 - 0.5) * 0.01 * 0.1 / norm0,
                        (2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                        - (1 - 0.5) * 0.01 * 0.1 / norm0,
                    ]
                ),
                self.evaluate(var0),
            )
            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        (3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                        - (1 - 0.5) * 0.01 * 0.01 / norm1,
                        (4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                        - (1 - 0.5) * 0.01 * 0.01 / norm1,
                    ]
                ),
                self.evaluate(var1),
            )

    def doTestBasicNuclear(self, use_resource=False, use_callable_params=False):
        # for i, dtype in enumerate([tf.half, tf.float32, tf.float64]):
        for i, dtype in enumerate([tf.float32, tf.float64]):
            # TODO:
            # Based on issue #36764 in the following link,
            #        "https://github.com/tensorflow/tensorflow/issues/36764"
            # tf.half is not registered for tf.linalg.svd function on Windows
            # CPU version.
            # So we have to remove tf.half when testing with Windows CPU version.

            if use_resource:
                var0 = tf.Variable([1.0, 2.0], dtype=dtype, name="var0_%d" % i)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype, name="var1_%d" % i)
            else:
                var0 = tf.Variable([1.0, 2.0], dtype=dtype)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype)

            grads0 = tf.constant([0.1, 0.1], dtype=dtype)
            grads1 = tf.constant([0.01, 0.01], dtype=dtype)
            top_singular_vector0 = self.top_singular_vector(grads0)
            top_singular_vector1 = self.top_singular_vector(grads1)

            def learning_rate():
                return 0.5

            def lambda_():
                return 0.01

            ord = "nuclear"

            if not use_callable_params:
                learning_rate = learning_rate()
                lambda_ = lambda_()

            cg_opt = cg_lib.ConditionalGradient(
                learning_rate=learning_rate, lambda_=lambda_, ord=ord
            )
            cg_update = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            if not tf.executing_eagerly():
                self.evaluate(tf.compat.v1.global_variables_initializer())
                # Fetch params to validate initial values
                self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                self.assertAllClose([3.0, 4.0], self.evaluate(var1))

            # Check we have slots
            self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
            slot0 = cg_opt.get_slot(var0, "conditional_gradient")
            self.assertEquals(slot0.get_shape(), var0.get_shape())
            slot1 = cg_opt.get_slot(var1, "conditional_gradient")
            self.assertEquals(slot1.get_shape(), var1.get_shape())

            if not tf.executing_eagerly():
                self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                self.assertFalse(slot1 in tf.compat.v1.trainable_variables())

            if not tf.executing_eagerly():
                self.evaluate(cg_update)

            top_singular_vector0 = self.evaluate(top_singular_vector0)
            top_singular_vector1 = self.evaluate(top_singular_vector1)

            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                        2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                    ]
                ),
                self.evaluate(var0),
            )
            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0],
                        4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                    ]
                ),
                self.evaluate(var1),
            )

            # Step 2: the conditional_gradient contain the previous update.
            if tf.executing_eagerly():
                cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            else:
                self.evaluate(cg_update)
            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        (1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0]) * 0.5
                        - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                        (2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1]) * 0.5
                        - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                    ]
                ),
                self.evaluate(var0),
            )
            self.assertAllCloseAccordingToType(
                np.array(
                    [
                        (3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0]) * 0.5
                        - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                        (4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0]) * 0.5
                        - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                    ]
                ),
                self.evaluate(var1),
            )

    def testBasicFrobenius(self):
        with self.cached_session():
            self.doTestBasicFrobenius(use_resource=False)

    def testResourceBasicFrobenius(self):
        self.doTestBasicFrobenius(use_resource=True)

    def testBasicCallableParamsFrobenius(self):
        self.doTestBasicFrobenius(use_resource=True, use_callable_params=True)

    def testBasicNuclear(self):
        with self.cached_session():
            self.doTestBasicNuclear(use_resource=False)

    def testResourceBasicNuclear(self):
        self.doTestBasicNuclear(use_resource=True)

    def testBasicCallableParamsNuclear(self):
        self.doTestBasicNuclear(use_resource=True, use_callable_params=True)

    def testVariablesAcrossGraphsFrobenius(self):
        optimizer = cg_lib.ConditionalGradient(0.01, 0.5, ord="fro")
        with tf.Graph().as_default():
            var0 = tf.Variable([1.0, 2.0], dtype=tf.float32, name="var0")
            var1 = tf.Variable([3.0, 4.0], dtype=tf.float32, name="var1")

            def loss():
                return tf.math.reduce_sum(var0 + var1)

            optimizer.minimize(loss, var_list=[var0, var1])
            optimizer_variables = optimizer.variables()
            # There should be three items. The first item is iteration,
            # and one item for each variable.
            self.assertStartsWith(
                optimizer_variables[1].name, "ConditionalGradient/var0"
            )
            self.assertStartsWith(
                optimizer_variables[2].name, "ConditionalGradient/var1"
            )
            self.assertEqual(3, len(optimizer_variables))

    def testVariablesAcrossGraphsNuclear(self):
        optimizer = cg_lib.ConditionalGradient(0.01, 0.5, ord="nuclear")
        with tf.Graph().as_default():
            var0 = tf.Variable([1.0, 2.0], dtype=tf.float32, name="var0")
            var1 = tf.Variable([3.0, 4.0], dtype=tf.float32, name="var1")

            def loss():
                return tf.math.reduce_sum(var0 + var1)

            optimizer.minimize(loss, var_list=[var0, var1])
            optimizer_variables = optimizer.variables()
            # There should be three items. The first item is iteration,
            # and one item for each variable.
            self.assertStartsWith(
                optimizer_variables[1].name, "ConditionalGradient/var0"
            )
            self.assertStartsWith(
                optimizer_variables[2].name, "ConditionalGradient/var1"
            )
            self.assertEqual(3, len(optimizer_variables))

    # Based on issue #347 in the following link,
    #        "https://github.com/tensorflow/addons/issues/347"
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel
    # for 'GPU' devices.
    # So we have to remove tf.half when testing with gpu.
    # The function "_DtypesToTest" is from
    #       "https://github.com/tensorflow/tensorflow/blob/5d4a6cee737a1dc6c20172a1dc1
    #        5df10def2df72/tensorflow/python/kernel_tests/conv_ops_3d_test.py#L53-L62"

    def _DtypesToTest(self, use_gpu):
        if use_gpu:
            return [tf.float32, tf.float64]
        else:
            return [tf.half, tf.float32, tf.float64]

    def testMinimizeSparseResourceVariableFrobenius(self):
        # This test invokes the ResourceSparseApplyConditionalGradient
        # operation. And it will call the 'ResourceScatterUpdate' OpKernel
        # for 'GPU' devices. However, tf.half is not registered in this case,
        # based on issue #347.
        # Thus, we will call the "_DtypesToTest" function.
        #
        # TODO:
        #       Wait for the solving of issue #347. After that, we will test
        #       for the dtype to be tf.half, with 'GPU' devices.
        for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
            var0 = tf.Variable([[1.0, 2.0]], dtype=dtype)

            def loss():
                x = tf.constant([[4.0], [5.0]], dtype=dtype)
                pred = tf.matmul(tf.nn.embedding_lookup([var0], [0]), x)
                return pred * pred

            # the gradient based on the current loss function
            grads0_0 = 32 * 1.0 + 40 * 2.0
            grads0_1 = 40 * 1.0 + 50 * 2.0
            grads0 = tf.constant([[grads0_0, grads0_1]], dtype=dtype)
            norm0 = tf.math.reduce_sum(grads0 ** 2) ** 0.5

            learning_rate = 0.1
            lambda_ = 0.1
            ord = "fro"
            opt = cg_lib.ConditionalGradient(
                learning_rate=learning_rate, lambda_=lambda_, ord=ord
            )
            cg_op = opt.minimize(loss, var_list=[var0])
            self.evaluate(tf.compat.v1.global_variables_initializer())

            # Run 1 step of cg_op
            self.evaluate(cg_op)

            # Validate updated params
            norm0 = self.evaluate(norm0)
            self.assertAllCloseAccordingToType(
                [
                    [
                        1.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * grads0_0 / norm0,
                        2.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * grads0_1 / norm0,
                    ]
                ],
                self.evaluate(var0),
            )

    def testMinimizeSparseResourceVariableNuclear(self):
        # This test invokes the ResourceSparseApplyConditionalGradient
        # operation. And it will call the 'ResourceScatterUpdate' OpKernel
        # for 'GPU' devices. However, tf.half is not registered in this case,
        # based on issue #347.
        # Thus, we will call the "_DtypesToTest" function.
        #
        # TODO:
        #       Wait for the solving of issue #347. After that, we will test
        #       for the dtype to be tf.half, with 'GPU' devices.
        # for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
        for dtype in [tf.float32, tf.float64]:
            # TODO:
            # Based on issue #36764 in the following link,
            #        "https://github.com/tensorflow/tensorflow/issues/36764"
            # tf.half is not registered for tf.linalg.svd function on Windows
            # CPU version.
            # So we have to remove tf.half when testing with Windows CPU version.
            var0 = tf.Variable([[1.0, 2.0]], dtype=dtype)

            def loss():
                x = tf.constant([[4.0], [5.0]], dtype=dtype)
                pred = tf.matmul(tf.nn.embedding_lookup([var0], [0]), x)
                return pred * pred

            # the gradient based on the current loss function
            grads0_0 = 32 * 1.0 + 40 * 2.0
            grads0_1 = 40 * 1.0 + 50 * 2.0
            grads0 = tf.constant([[grads0_0, grads0_1]], dtype=dtype)
            top_singular_vector0 = self.top_singular_vector(grads0)

            learning_rate = 0.1
            lambda_ = 0.1
            ord = "nuclear"
            opt = cg_lib.ConditionalGradient(
                learning_rate=learning_rate, lambda_=lambda_, ord=ord
            )
            cg_op = opt.minimize(loss, var_list=[var0])
            self.evaluate(tf.compat.v1.global_variables_initializer())

            # Run 1 step of cg_op
            self.evaluate(cg_op)

            # Validate updated params
            top_singular_vector0 = self.evaluate(top_singular_vector0)
            self.assertAllCloseAccordingToType(
                [
                    [
                        1.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector0[0][0],
                        2.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector0[0][1],
                    ]
                ],
                self.evaluate(var0),
            )

    def testMinimizeWith2DIndiciesForEmbeddingLookupFrobenius(self):
        # This test invokes the ResourceSparseApplyConditionalGradient
        # operation.
        var0 = tf.Variable(tf.ones([2, 2]))

        def loss():
            return tf.math.reduce_sum(tf.nn.embedding_lookup(var0, [[1]]))

        # the gradient for this loss function:
        grads0 = tf.constant([[0, 0], [1, 1]], dtype=tf.float32)
        norm0 = tf.math.reduce_sum(grads0 ** 2) ** 0.5

        learning_rate = 0.1
        lambda_ = 0.1
        ord = "fro"
        opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        cg_op = opt.minimize(loss, var_list=[var0])
        self.evaluate(tf.compat.v1.global_variables_initializer())

        # Run 1 step of cg_op
        self.evaluate(cg_op)
        norm0 = self.evaluate(norm0)
        self.assertAllCloseAccordingToType(
            [
                [1, 1],
                [
                    learning_rate * 1 - (1 - learning_rate) * lambda_ * 1 / norm0,
                    learning_rate * 1 - (1 - learning_rate) * lambda_ * 1 / norm0,
                ],
            ],
            self.evaluate(var0),
        )

    def testMinimizeWith2DIndiciesForEmbeddingLookupNuclear(self):
        # This test invokes the ResourceSparseApplyConditionalGradient
        # operation.
        var0 = tf.Variable(tf.ones([2, 2]))

        def loss():
            return tf.math.reduce_sum(tf.nn.embedding_lookup(var0, [[1]]))

        # the gradient for this loss function:
        grads0 = tf.constant([[0, 0], [1, 1]], dtype=tf.float32)
        top_singular_vector0 = self.top_singular_vector(grads0)

        learning_rate = 0.1
        lambda_ = 0.1
        ord = "nuclear"
        opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        cg_op = opt.minimize(loss, var_list=[var0])
        self.evaluate(tf.compat.v1.global_variables_initializer())

        # Run 1 step of cg_op
        self.evaluate(cg_op)
        top_singular_vector0 = self.evaluate(top_singular_vector0)
        self.evaluate(var0)
        self.assertAllCloseAccordingToType(
            [
                learning_rate * 1
                - (1 - learning_rate) * lambda_ * top_singular_vector0[1][0],
                learning_rate * 1
                - (1 - learning_rate) * lambda_ * top_singular_vector0[1][1],
            ],
            self.evaluate(var0[1]),
        )

    def testTensorLearningRateAndConditionalGradientFrobenius(self):
        for dtype in [tf.half, tf.float32, tf.float64]:
            with self.cached_session():
                var0 = tf.Variable([1.0, 2.0], dtype=dtype)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype)
                grads0 = tf.constant([0.1, 0.1], dtype=dtype)
                grads1 = tf.constant([0.01, 0.01], dtype=dtype)
                norm0 = tf.math.reduce_sum(grads0 ** 2) ** 0.5
                norm1 = tf.math.reduce_sum(grads1 ** 2) ** 0.5
                ord = "fro"
                cg_opt = cg_lib.ConditionalGradient(
                    learning_rate=tf.constant(0.5), lambda_=tf.constant(0.01), ord=ord
                )
                cg_update = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                if not tf.executing_eagerly():
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    # Fetch params to validate initial values
                    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

                # Check we have slots
                self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
                slot0 = cg_opt.get_slot(var0, "conditional_gradient")
                self.assertEquals(slot0.get_shape(), var0.get_shape())
                slot1 = cg_opt.get_slot(var1, "conditional_gradient")
                self.assertEquals(slot1.get_shape(), var1.get_shape())

                if not tf.executing_eagerly():
                    self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                    self.assertFalse(slot1 in tf.compat.v1.trainable_variables())

                if not tf.executing_eagerly():
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                norm0 = self.evaluate(norm0)
                norm1 = self.evaluate(norm1)
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
                            2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
                        ]
                    ),
                    self.evaluate(var0),
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
                            4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
                        ]
                    ),
                    self.evaluate(var1),
                )
                # Step 2: the conditional_gradient contain the
                # previous update.
                if tf.executing_eagerly():
                    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                else:
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                            - (1 - 0.5) * 0.01 * 0.1 / norm0,
                            (2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                            - (1 - 0.5) * 0.01 * 0.1 / norm0,
                        ]
                    ),
                    self.evaluate(var0),
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                            - (1 - 0.5) * 0.01 * 0.01 / norm1,
                            (4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                            - (1 - 0.5) * 0.01 * 0.01 / norm1,
                        ]
                    ),
                    self.evaluate(var1),
                )

    def testTensorLearningRateAndConditionalGradientNuclear(self):
        # for dtype in [tf.half, tf.float32, tf.float64]:
        for dtype in [tf.float32, tf.float64]:
            # TODO:
            # Based on issue #36764 in the following link,
            #        "https://github.com/tensorflow/tensorflow/issues/36764"
            # tf.half is not registered for tf.linalg.svd function on Windows
            # CPU version.
            # So we have to remove tf.half when testing with Windows CPU version.
            with self.cached_session():
                var0 = tf.Variable([1.0, 2.0], dtype=dtype)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype)
                grads0 = tf.constant([0.1, 0.1], dtype=dtype)
                grads1 = tf.constant([0.01, 0.01], dtype=dtype)
                top_singular_vector0 = self.top_singular_vector(grads0)
                top_singular_vector1 = self.top_singular_vector(grads1)
                ord = "nuclear"
                cg_opt = cg_lib.ConditionalGradient(
                    learning_rate=tf.constant(0.5), lambda_=tf.constant(0.01), ord=ord
                )
                cg_update = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                if not tf.executing_eagerly():
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    # Fetch params to validate initial values
                    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

                # Check we have slots
                self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
                slot0 = cg_opt.get_slot(var0, "conditional_gradient")
                self.assertEquals(slot0.get_shape(), var0.get_shape())
                slot1 = cg_opt.get_slot(var1, "conditional_gradient")
                self.assertEquals(slot1.get_shape(), var1.get_shape())

                if not tf.executing_eagerly():
                    self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                    self.assertFalse(slot1 in tf.compat.v1.trainable_variables())

                if not tf.executing_eagerly():
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                top_singular_vector0 = self.evaluate(top_singular_vector0)
                top_singular_vector1 = self.evaluate(top_singular_vector1)
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                            2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                        ]
                    ),
                    self.evaluate(var0),
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0],
                            4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                        ]
                    ),
                    self.evaluate(var1),
                )
                # Step 2: the conditional_gradient contain the
                # previous update.
                if tf.executing_eagerly():
                    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                else:
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0])
                            * 0.5
                            - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                            (2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1])
                            * 0.5
                            - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                        ]
                    ),
                    self.evaluate(var0),
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0])
                            * 0.5
                            - (1 - 0.5) * 0.01 * top_singular_vector1[0],
                            (4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[1])
                            * 0.5
                            - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                        ]
                    ),
                    self.evaluate(var1),
                )

    def _dbParamsFrobeniusCG01(self):
        """Return dist-belief conditional_gradient values.

        Return values been generated from the dist-belief
        conditional_gradient unittest, running with a learning rate of 0.1
        and a lambda_ of 0.1.

        These values record how a parameter vector of size 10, initialized
        with 0.0, gets updated with 10 consecutive conditional_gradient
        steps.
        It uses random gradients.

        Returns:
            db_grad: The gradients to apply
            db_out: The parameters after the conditional_gradient update.
        """
        db_grad = [[]] * 10
        db_out = [[]] * 10
        db_grad[0] = [
            0.00096264342,
            0.17914793,
            0.93945462,
            0.41396621,
            0.53037018,
            0.93197989,
            0.78648776,
            0.50036013,
            0.55345792,
            0.96722615,
        ]
        db_out[0] = [
            -4.1555551e-05,
            -7.7334875e-03,
            -4.0554531e-02,
            -1.7870162e-02,
            -2.2895107e-02,
            -4.0231861e-02,
            -3.3951234e-02,
            -2.1599628e-02,
            -2.3891762e-02,
            -4.1753378e-02,
        ]
        db_grad[1] = [
            0.17075552,
            0.88821375,
            0.20873757,
            0.25236958,
            0.57578111,
            0.15312378,
            0.5513742,
            0.94687688,
            0.16012503,
            0.22159521,
        ]
        db_out[1] = [
            -0.00961733,
            -0.0507779,
            -0.01580694,
            -0.01599489,
            -0.03470477,
            -0.01264373,
            -0.03443632,
            -0.05546713,
            -0.01140388,
            -0.01665068,
        ]
        db_grad[2] = [
            0.35077485,
            0.47304362,
            0.44412705,
            0.44368884,
            0.078527533,
            0.81223965,
            0.31168157,
            0.43203235,
            0.16792089,
            0.24644311,
        ]
        db_out[2] = [
            -0.02462724,
            -0.03699233,
            -0.03154434,
            -0.03153357,
            -0.00876844,
            -0.05606323,
            -0.02447166,
            -0.03469437,
            -0.0124694,
            -0.01829169,
        ]
        db_grad[3] = [
            0.9694621,
            0.75035888,
            0.28171822,
            0.83813518,
            0.53807181,
            0.3728098,
            0.81454384,
            0.03848977,
            0.89759839,
            0.93665648,
        ]
        db_out[3] = [
            -0.04124615,
            -0.03371741,
            -0.0144246,
            -0.03668303,
            -0.02240246,
            -0.02052062,
            -0.03503307,
            -0.00500922,
            -0.03715545,
            -0.0393002,
        ]
        db_grad[4] = [
            0.38578293,
            0.8536852,
            0.88722926,
            0.66276771,
            0.13678469,
            0.94036359,
            0.69107032,
            0.81897682,
            0.5433259,
            0.67860287,
        ]
        db_out[4] = [
            -0.01979208,
            -0.0380417,
            -0.03747472,
            -0.0305847,
            -0.00779536,
            -0.04024222,
            -0.03156913,
            -0.0337613,
            -0.02578116,
            -0.03148952,
        ]
        db_grad[5] = [
            0.27885768,
            0.76100707,
            0.24625534,
            0.81354135,
            0.18959245,
            0.48038563,
            0.84163809,
            0.41172323,
            0.83259648,
            0.44941229,
        ]
        db_out[5] = [
            -0.01555188,
            -0.04084422,
            -0.01573331,
            -0.04265549,
            -0.01000746,
            -0.02740575,
            -0.04412147,
            -0.02341569,
            -0.0431026,
            -0.02502293,
        ]
        db_grad[6] = [
            0.27233034,
            0.056316052,
            0.5039115,
            0.24105175,
            0.35697976,
            0.75913221,
            0.73577434,
            0.16014607,
            0.57500273,
            0.071136251,
        ]
        db_out[6] = [
            -0.01890448,
            -0.00767214,
            -0.03367592,
            -0.01962219,
            -0.02374279,
            -0.05110247,
            -0.05128598,
            -0.01254396,
            -0.04094185,
            -0.00703416,
        ]
        db_grad[7] = [
            0.58697265,
            0.2494842,
            0.08106143,
            0.39954534,
            0.15892942,
            0.12683646,
            0.74053431,
            0.16033,
            0.66625422,
            0.73515922,
        ]
        db_out[7] = [
            -0.03772914,
            -0.01599993,
            -0.00831695,
            -0.02635719,
            -0.01207801,
            -0.01285448,
            -0.05034328,
            -0.01104364,
            -0.04477356,
            -0.04558991,
        ]
        db_grad[8] = [
            0.8215279,
            0.41994119,
            0.95172721,
            0.68000203,
            0.79439718,
            0.43384039,
            0.55561525,
            0.22567581,
            0.93331909,
            0.29438227,
        ]
        db_out[8] = [
            -0.03919835,
            -0.01970845,
            -0.04187151,
            -0.03195836,
            -0.03546333,
            -0.01999326,
            -0.02899324,
            -0.01083582,
            -0.04472339,
            -0.01725317,
        ]
        db_grad[9] = [
            0.68297005,
            0.67758518,
            0.1748755,
            0.13266537,
            0.70697063,
            0.055731893,
            0.68593478,
            0.50580865,
            0.12602448,
            0.093537711,
        ]
        db_out[9] = [
            -0.04510314,
            -0.04282944,
            -0.0147322,
            -0.0111956,
            -0.04617687,
            -0.00535998,
            -0.0442614,
            -0.03158399,
            -0.01207165,
            -0.00736567,
        ]
        return db_grad, db_out

    def testLikeDistBeliefFrobeniusCG01(self):
        with self.cached_session():
            db_grad, db_out = self._dbParamsFrobeniusCG01()
            num_samples = len(db_grad)
            var0 = tf.Variable([0.0] * num_samples)
            grads0 = tf.constant([0.0] * num_samples)
            ord = "fro"
            cg_opt = cg_lib.ConditionalGradient(learning_rate=0.1, lambda_=0.1, ord=ord)
            if not tf.executing_eagerly():
                cg_update = cg_opt.apply_gradients(zip([grads0], [var0]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            for i in range(num_samples):
                if tf.executing_eagerly():
                    grads0 = tf.constant(db_grad[i])
                    cg_opt.apply_gradients(zip([grads0], [var0]))
                else:
                    cg_update.run(feed_dict={grads0: db_grad[i]})
                self.assertAllClose(np.array(db_out[i]), self.evaluate(var0))

    def _dbParamsNuclearCG01(self):
        """Return dist-belief conditional_gradient values.

        Return values been generated from the dist-belief
        conditional_gradient unittest, running with a learning rate of 0.1
        and a lambda_ of 0.1.

        These values record how a parameter vector of size 10, initialized
        with 0.0, gets updated with 10 consecutive conditional_gradient
        steps.
        It uses random gradients.

        Returns:
            db_grad: The gradients to apply
            db_out: The parameters after the conditional_gradient update.
        """
        db_grad = [[]] * 10
        db_out = [[]] * 10
        db_grad[0] = [
            0.00096264342,
            0.17914793,
            0.93945462,
            0.41396621,
            0.53037018,
            0.93197989,
            0.78648776,
            0.50036013,
            0.55345792,
            0.96722615,
        ]
        db_out[0] = [
            -4.1552783e-05,
            -7.7334875e-03,
            -4.0554535e-02,
            -1.7870164e-02,
            -2.2895109e-02,
            -4.0231861e-02,
            -3.3951234e-02,
            -2.1599628e-02,
            -2.3891764e-02,
            -4.1753381e-02,
        ]
        db_grad[1] = [
            0.17075552,
            0.88821375,
            0.20873757,
            0.25236958,
            0.57578111,
            0.15312378,
            0.5513742,
            0.94687688,
            0.16012503,
            0.22159521,
        ]
        db_out[1] = [
            -0.00961733,
            -0.0507779,
            -0.01580694,
            -0.01599489,
            -0.03470477,
            -0.01264373,
            -0.03443632,
            -0.05546713,
            -0.01140388,
            -0.01665068,
        ]
        db_grad[2] = [
            0.35077485,
            0.47304362,
            0.44412705,
            0.44368884,
            0.078527533,
            0.81223965,
            0.31168157,
            0.43203235,
            0.16792089,
            0.24644311,
        ]
        db_out[2] = [
            -0.02462724,
            -0.03699233,
            -0.03154433,
            -0.03153357,
            -0.00876844,
            -0.05606324,
            -0.02447166,
            -0.03469437,
            -0.0124694,
            -0.01829169,
        ]
        db_grad[3] = [
            0.9694621,
            0.75035888,
            0.28171822,
            0.83813518,
            0.53807181,
            0.3728098,
            0.81454384,
            0.03848977,
            0.89759839,
            0.93665648,
        ]
        db_out[3] = [
            -0.04124615,
            -0.03371741,
            -0.0144246,
            -0.03668303,
            -0.02240246,
            -0.02052062,
            -0.03503307,
            -0.00500922,
            -0.03715545,
            -0.0393002,
        ]
        db_grad[4] = [
            0.38578293,
            0.8536852,
            0.88722926,
            0.66276771,
            0.13678469,
            0.94036359,
            0.69107032,
            0.81897682,
            0.5433259,
            0.67860287,
        ]
        db_out[4] = [
            -0.01979207,
            -0.0380417,
            -0.03747472,
            -0.0305847,
            -0.00779536,
            -0.04024221,
            -0.03156913,
            -0.0337613,
            -0.02578116,
            -0.03148951,
        ]
        db_grad[5] = [
            0.27885768,
            0.76100707,
            0.24625534,
            0.81354135,
            0.18959245,
            0.48038563,
            0.84163809,
            0.41172323,
            0.83259648,
            0.44941229,
        ]
        db_out[5] = [
            -0.01555188,
            -0.04084422,
            -0.01573331,
            -0.04265549,
            -0.01000746,
            -0.02740575,
            -0.04412147,
            -0.02341569,
            -0.0431026,
            -0.02502293,
        ]
        db_grad[6] = [
            0.27233034,
            0.056316052,
            0.5039115,
            0.24105175,
            0.35697976,
            0.75913221,
            0.73577434,
            0.16014607,
            0.57500273,
            0.071136251,
        ]
        db_out[6] = [
            -0.01890448,
            -0.00767214,
            -0.03367592,
            -0.01962219,
            -0.02374278,
            -0.05110246,
            -0.05128598,
            -0.01254396,
            -0.04094184,
            -0.00703416,
        ]
        db_grad[7] = [
            0.58697265,
            0.2494842,
            0.08106143,
            0.39954534,
            0.15892942,
            0.12683646,
            0.74053431,
            0.16033,
            0.66625422,
            0.73515922,
        ]
        db_out[7] = [
            -0.03772915,
            -0.01599993,
            -0.00831695,
            -0.0263572,
            -0.01207801,
            -0.01285448,
            -0.05034329,
            -0.01104364,
            -0.04477356,
            -0.04558992,
        ]
        db_grad[8] = [
            0.8215279,
            0.41994119,
            0.95172721,
            0.68000203,
            0.79439718,
            0.43384039,
            0.55561525,
            0.22567581,
            0.93331909,
            0.29438227,
        ]
        db_out[8] = [
            -0.03919835,
            -0.01970845,
            -0.04187151,
            -0.03195836,
            -0.03546333,
            -0.01999326,
            -0.02899324,
            -0.01083582,
            -0.04472339,
            -0.01725317,
        ]
        db_grad[9] = [
            0.68297005,
            0.67758518,
            0.1748755,
            0.13266537,
            0.70697063,
            0.055731893,
            0.68593478,
            0.50580865,
            0.12602448,
            0.093537711,
        ]
        db_out[9] = [
            -0.04510314,
            -0.04282944,
            -0.0147322,
            -0.0111956,
            -0.04617687,
            -0.00535998,
            -0.0442614,
            -0.031584,
            -0.01207165,
            -0.00736567,
        ]
        return db_grad, db_out

    def testLikeDistBeliefNuclearCG01(self):
        with self.cached_session():
            db_grad, db_out = self._dbParamsNuclearCG01()
            num_samples = len(db_grad)
            var0 = tf.Variable([0.0] * num_samples)
            grads0 = tf.constant([0.0] * num_samples)
            ord = "nuclear"
            cg_opt = cg_lib.ConditionalGradient(learning_rate=0.1, lambda_=0.1, ord=ord)
            if not tf.executing_eagerly():
                cg_update = cg_opt.apply_gradients(zip([grads0], [var0]))
                self.evaluate(tf.compat.v1.global_variables_initializer())

            for i in range(num_samples):
                if tf.executing_eagerly():
                    grads0 = tf.constant(db_grad[i])
                    cg_opt.apply_gradients(zip([grads0], [var0]))
                else:
                    cg_update.run(feed_dict={grads0: db_grad[i]})
                self.assertAllClose(np.array(db_out[i]), self.evaluate(var0))

    def testSparseFrobenius(self):
        # TODO:
        #       To address the issue #347.
        for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
            with self.cached_session():
                var0 = tf.Variable(tf.zeros([4, 2], dtype=dtype))
                var1 = tf.Variable(tf.constant(1.0, dtype, [4, 2]))
                grads0 = tf.IndexedSlices(
                    tf.constant([[0.1, 0.1]], dtype=dtype),
                    tf.constant([1]),
                    tf.constant([4, 2]),
                )
                grads1 = tf.IndexedSlices(
                    tf.constant([[0.01, 0.01], [0.01, 0.01]], dtype=dtype),
                    tf.constant([2, 3]),
                    tf.constant([4, 2]),
                )
                norm0 = tf.math.reduce_sum(tf.math.multiply(grads0, grads0)) ** 0.5
                norm1 = tf.math.reduce_sum(tf.math.multiply(grads1, grads1)) ** 0.5
                learning_rate = 0.1
                lambda_ = 0.1
                ord = "fro"
                cg_opt = cg_lib.ConditionalGradient(
                    learning_rate=learning_rate, lambda_=lambda_, ord=ord
                )
                cg_update = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                if not tf.executing_eagerly():
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    # Fetch params to validate initial values
                    self.assertAllClose([0, 0], self.evaluate(var0)[0])
                    self.assertAllClose([0, 0], self.evaluate(var0)[1])
                    self.assertAllClose([1, 1], self.evaluate(var1)[2])

                # Check we have slots
                self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
                slot0 = cg_opt.get_slot(var0, "conditional_gradient")
                self.assertEquals(slot0.get_shape(), var0.get_shape())
                slot1 = cg_opt.get_slot(var1, "conditional_gradient")
                self.assertEquals(slot1.get_shape(), var1.get_shape())

                if not tf.executing_eagerly():
                    self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                    self.assertFalse(slot1 in tf.compat.v1.trainable_variables())

                # Step 1:
                if not tf.executing_eagerly():
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                norm0 = self.evaluate(norm0)
                norm1 = self.evaluate(norm1)
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            0 - (1 - learning_rate) * lambda_ * 0 / norm0,
                            0 - (1 - learning_rate) * lambda_ * 0 / norm0,
                        ]
                    ),
                    self.evaluate(var0)[0],
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            0 - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                            0 - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                        ]
                    ),
                    self.evaluate(var0)[1],
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            1.0 * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                            1.0 * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                        ]
                    ),
                    self.evaluate(var1)[2],
                )
                # Step 2: the conditional_gradient contain the
                # previous update.
                if tf.executing_eagerly():
                    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                else:
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                self.assertAllClose(np.array([0, 0]), self.evaluate(var0)[0])
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (0 - (1 - learning_rate) * lambda_ * 0.1 / norm0)
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                            (0 - (1 - learning_rate) * lambda_ * 0.1 / norm0)
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                        ]
                    ),
                    self.evaluate(var0)[1],
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                1.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.01 / norm1
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                            (
                                1.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.01 / norm1
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                        ]
                    ),
                    self.evaluate(var1)[2],
                )

    def testSparseNuclear(self):
        # TODO:
        #       To address the issue #347.
        # for dtype in self._DtypesToTest(use_gpu=tf.test.is_gpu_available()):
        for dtype in [tf.float32, tf.float64]:
            # TODO:
            # Based on issue #36764 in the following link,
            #        "https://github.com/tensorflow/tensorflow/issues/36764"
            # tf.half is not registered for tf.linalg.svd function on Windows
            # CPU version.
            # So we have to remove tf.half when testing with Windows CPU version.
            with self.cached_session():
                var0 = tf.Variable(tf.zeros([4, 2], dtype=dtype))
                var1 = tf.Variable(tf.constant(1.0, dtype, [4, 2]))
                grads0 = tf.IndexedSlices(
                    tf.constant([[0.1, 0.1]], dtype=dtype),
                    tf.constant([1]),
                    tf.constant([4, 2]),
                )
                grads1 = tf.IndexedSlices(
                    tf.constant([[0.01, 0.01], [0.01, 0.01]], dtype=dtype),
                    tf.constant([2, 3]),
                    tf.constant([4, 2]),
                )
                top_singular_vector0 = tf.constant(
                    [[0.0, 0.0], [0.7071067, 0.7071067], [0.0, 0.0], [0.0, 0.0]],
                    dtype=dtype,
                )
                top_singular_vector1 = tf.constant(
                    [
                        [-4.2146844e-08, -4.2146844e-08],
                        [0.0000000e00, 0.0000000e00],
                        [4.9999994e-01, 4.9999994e-01],
                        [4.9999994e-01, 4.9999994e-01],
                    ],
                    dtype=dtype,
                )
                learning_rate = 0.1
                lambda_ = 0.1
                ord = "nuclear"
                cg_opt = cg_lib.ConditionalGradient(
                    learning_rate=learning_rate, lambda_=lambda_, ord=ord
                )
                cg_update = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

                if not tf.executing_eagerly():
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    # Fetch params to validate initial values
                    self.assertAllClose([0, 0], self.evaluate(var0)[0])
                    self.assertAllClose([0, 0], self.evaluate(var0)[1])
                    self.assertAllClose([1, 1], self.evaluate(var1)[2])
                # Check we have slots
                self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
                slot0 = cg_opt.get_slot(var0, "conditional_gradient")
                self.assertEquals(slot0.get_shape(), var0.get_shape())
                slot1 = cg_opt.get_slot(var1, "conditional_gradient")
                self.assertEquals(slot1.get_shape(), var1.get_shape())
                if not tf.executing_eagerly():
                    self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                    self.assertFalse(slot1 in tf.compat.v1.trainable_variables())

                # Step 1:
                if not tf.executing_eagerly():
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.

                top_singular_vector0 = self.evaluate(top_singular_vector0)
                top_singular_vector1 = self.evaluate(top_singular_vector1)

                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            0
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector0[0][0],
                            0
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector0[0][1],
                        ]
                    ),
                    self.evaluate(var0)[0],
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            0
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector0[1][0],
                            0
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector0[1][1],
                        ]
                    ),
                    self.evaluate(var0)[1],
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            1.0 * learning_rate
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector1[2][0],
                            1.0 * learning_rate
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector1[2][1],
                        ]
                    ),
                    self.evaluate(var1)[2],
                )
                # Step 2: the conditional_gradient contain the
                # previous update.
                if tf.executing_eagerly():
                    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                else:
                    self.evaluate(cg_update)
                # Check that the parameters have been updated.
                self.assertAllClose(np.array([0, 0]), self.evaluate(var0)[0])
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                0
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector0[1][0]
                            )
                            * learning_rate
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector0[1][0],
                            (
                                0
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector0[1][1]
                            )
                            * learning_rate
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector0[1][1],
                        ]
                    ),
                    self.evaluate(var0)[1],
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                1.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector1[2][0]
                            )
                            * learning_rate
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector1[2][0],
                            (
                                1.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector1[2][1]
                            )
                            * learning_rate
                            - (1 - learning_rate)
                            * lambda_
                            * top_singular_vector1[2][1],
                        ]
                    ),
                    self.evaluate(var1)[2],
                )

    def testSharingFrobenius(self):
        for dtype in [tf.half, tf.float32, tf.float64]:
            with self.cached_session():
                var0 = tf.Variable([1.0, 2.0], dtype=dtype)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype)
                grads0 = tf.constant([0.1, 0.1], dtype=dtype)
                grads1 = tf.constant([0.01, 0.01], dtype=dtype)
                norm0 = tf.math.reduce_sum(grads0 ** 2) ** 0.5
                norm1 = tf.math.reduce_sum(grads1 ** 2) ** 0.5
                learning_rate = 0.1
                lambda_ = 0.1
                ord = "fro"
                cg_opt = cg_lib.ConditionalGradient(
                    learning_rate=learning_rate, lambda_=lambda_, ord=ord
                )
                cg_update1 = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                cg_update2 = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                if not tf.executing_eagerly():
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    # Fetch params to validate initial values
                    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

                # Check we have slots
                self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
                slot0 = cg_opt.get_slot(var0, "conditional_gradient")
                self.assertEquals(slot0.get_shape(), var0.get_shape())
                slot1 = cg_opt.get_slot(var1, "conditional_gradient")
                self.assertEquals(slot1.get_shape(), var1.get_shape())

                if not tf.executing_eagerly():
                    self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                    self.assertFalse(slot1 in tf.compat.v1.trainable_variables())
                # Because in the eager mode, as we declare two cg_update
                # variables, it already altomatically finish executing them.
                # Thus, we cannot test the param value at this time for
                # eager mode. We can only test the final value of param
                # after the second execution.
                if not tf.executing_eagerly():
                    self.evaluate(cg_update1)
                    # Check that the parameters have been updated.
                    norm0 = self.evaluate(norm0)
                    norm1 = self.evaluate(norm1)
                    self.assertAllCloseAccordingToType(
                        np.array(
                            [
                                1.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                                2.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                            ]
                        ),
                        self.evaluate(var0),
                    )
                    self.assertAllCloseAccordingToType(
                        np.array(
                            [
                                3.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                                4.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                            ]
                        ),
                        self.evaluate(var1),
                    )

                # Step 2: the second conditional_gradient contain
                # the previous update.
                if not tf.executing_eagerly():
                    self.evaluate(cg_update2)
                # Check that the parameters have been updated.
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                1.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.1 / norm0
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                            (
                                2.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.1 / norm0
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                        ]
                    ),
                    self.evaluate(var0),
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                3.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.01 / norm1
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                            (
                                4.0 * learning_rate
                                - (1 - learning_rate) * lambda_ * 0.01 / norm1
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                        ]
                    ),
                    self.evaluate(var1),
                )

    def testSharingNuclear(self):
        # for dtype in [tf.half, tf.float32, tf.float64]:
        for dtype in [tf.float32, tf.float64]:
            # TODO:
            # Based on issue #36764 in the following link,
            #        "https://github.com/tensorflow/tensorflow/issues/36764"
            # tf.half is not registered for tf.linalg.svd function on Windows
            # CPU version.
            # So we have to remove tf.half when testing with Windows CPU version.
            with self.cached_session():
                var0 = tf.Variable([1.0, 2.0], dtype=dtype)
                var1 = tf.Variable([3.0, 4.0], dtype=dtype)
                grads0 = tf.constant([0.1, 0.1], dtype=dtype)
                grads1 = tf.constant([0.01, 0.01], dtype=dtype)
                top_singular_vector0 = self.top_singular_vector(grads0)
                top_singular_vector1 = self.top_singular_vector(grads1)
                learning_rate = 0.1
                lambda_ = 0.1
                ord = "nuclear"
                cg_opt = cg_lib.ConditionalGradient(
                    learning_rate=learning_rate, lambda_=lambda_, ord=ord
                )
                cg_update1 = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                cg_update2 = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
                if not tf.executing_eagerly():
                    self.evaluate(tf.compat.v1.global_variables_initializer())
                    # Fetch params to validate initial values
                    self.assertAllClose([1.0, 2.0], self.evaluate(var0))
                    self.assertAllClose([3.0, 4.0], self.evaluate(var1))

                # Check we have slots
                self.assertEqual(["conditional_gradient"], cg_opt.get_slot_names())
                slot0 = cg_opt.get_slot(var0, "conditional_gradient")
                self.assertEquals(slot0.get_shape(), var0.get_shape())
                slot1 = cg_opt.get_slot(var1, "conditional_gradient")
                self.assertEquals(slot1.get_shape(), var1.get_shape())

                if not tf.executing_eagerly():
                    self.assertFalse(slot0 in tf.compat.v1.trainable_variables())
                    self.assertFalse(slot1 in tf.compat.v1.trainable_variables())
                # Because in the eager mode, as we declare two cg_update
                # variables, it already altomatically finish executing them.
                # Thus, we cannot test the param value at this time for
                # eager mode. We can only test the final value of param
                # after the second execution.
                if not tf.executing_eagerly():
                    self.evaluate(cg_update1)
                    # Check that the parameters have been updated.
                    top_singular_vector0 = self.evaluate(top_singular_vector0)
                    top_singular_vector1 = self.evaluate(top_singular_vector1)
                    self.assertAllCloseAccordingToType(
                        np.array(
                            [
                                1.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector0[0],
                                2.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector0[1],
                            ]
                        ),
                        self.evaluate(var0),
                    )
                    self.assertAllCloseAccordingToType(
                        np.array(
                            [
                                3.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector1[0],
                                4.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector1[1],
                            ]
                        ),
                        self.evaluate(var1),
                    )

                # Step 2: the second conditional_gradient contain
                # the previous update.
                if not tf.executing_eagerly():
                    self.evaluate(cg_update2)
                # Check that the parameters have been updated.
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                1.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector0[0]
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * top_singular_vector0[0],
                            (
                                2.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector0[1]
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * top_singular_vector0[1],
                        ]
                    ),
                    self.evaluate(var0),
                )
                self.assertAllCloseAccordingToType(
                    np.array(
                        [
                            (
                                3.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector1[0]
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * top_singular_vector1[0],
                            (
                                4.0 * learning_rate
                                - (1 - learning_rate)
                                * lambda_
                                * top_singular_vector1[1]
                            )
                            * learning_rate
                            - (1 - learning_rate) * lambda_ * top_singular_vector1[1],
                        ]
                    ),
                    self.evaluate(var1),
                )


if __name__ == "__main__":
    tf.test.main()
