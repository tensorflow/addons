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
"""Tests for Rectified Adam optimizer."""

import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead


@test_utils.run_all_in_graph_and_eager_modes
class RectifiedAdamTest(tf.test.TestCase):
    def run_dense_sample(self, iterations, expected, optimizer):
        var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
        var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

        grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
        grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        if tf.executing_eagerly():
            for _ in range(iterations):
                optimizer.apply_gradients(grads_and_vars)
        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        self.assertAllClose(var_0.read_value(), expected[0], atol=2e-4)
        self.assertAllClose(var_1.read_value(), expected[1], atol=2e-4)

    def run_sparse_sample(self, iterations, expected, optimizer):
        var_0 = tf.Variable([1.0, 2.0])
        var_1 = tf.Variable([3.0, 4.0])

        grad_0 = tf.IndexedSlices(
            tf.constant([0.1]), tf.constant([0]), tf.constant([2])
        )
        grad_1 = tf.IndexedSlices(
            tf.constant([0.04]), tf.constant([1]), tf.constant([2])
        )

        grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

        if tf.executing_eagerly():
            for _ in range(iterations):
                optimizer.apply_gradients(grads_and_vars)
        else:
            update = optimizer.apply_gradients(grads_and_vars)
            self.evaluate(tf.compat.v1.global_variables_initializer())
            for _ in range(iterations):
                self.evaluate(update)

        self.assertAllClose(var_0.read_value(), expected[0], atol=2e-4)
        self.assertAllClose(var_1.read_value(), expected[1], atol=2e-4)

    def test_dense_sample(self):
        # Expected values are obtained from the previous implementation
        self.run_dense_sample(
            iterations=100,
            expected=[[0.985769, 1.985269], [2.986119, 3.986068]],
            optimizer=RectifiedAdam(lr=1e-3),
        )

    def test_sparse_sample(self):
        # Expected values are obtained from the previous implementation
        self.run_sparse_sample(
            iterations=200,
            expected=[[0.959333, 2.0], [3.0, 3.959632]],
            optimizer=RectifiedAdam(lr=1e-3),
        )

    def test_dense_sample_with_amsgrad(self):
        # Expected values are obtained from the official implementation
        # `amsgrad` has no effect because the gradient is fixed
        self.run_dense_sample(
            iterations=100,
            expected=[[0.985769, 1.985269], [2.986119, 3.986068]],
            optimizer=RectifiedAdam(lr=1e-3, amsgrad=True),
        )

    def test_sparse_sample_with_amsgrad(self):
        # Expected values are obtained from the official implementation
        # `amsgrad` has no effect because the gradient is fixed
        self.run_sparse_sample(
            iterations=200,
            expected=[[0.959333, 2.0], [3.0, 3.959632]],
            optimizer=RectifiedAdam(lr=1e-3, amsgrad=True),
        )

    def test_dense_sample_with_weight_decay(self):
        # Expected values are obtained from the previous implementation
        self.run_dense_sample(
            iterations=100,
            expected=[[0.984775, 1.983276], [2.983125, 3.982076]],
            optimizer=RectifiedAdam(lr=1e-3, weight_decay=0.01),
        )

    def test_sparse_sample_with_weight_decay(self):
        # Expected values are obtained from the previous implementation
        self.run_sparse_sample(
            iterations=200,
            expected=[[0.957368, 2.0], [3.0, 3.951673]],
            optimizer=RectifiedAdam(lr=1e-3, weight_decay=0.01),
        )

    def test_dense_sample_with_warmup(self):
        self.run_dense_sample(
            iterations=100,
            expected=[[0.994062, 1.993912], [2.994167, 3.994152]],
            optimizer=RectifiedAdam(
                lr=1e-3, total_steps=100, warmup_proportion=0.1, min_lr=1e-5,
            ),
        )

    def test_sparse_sample_with_warmup(self):
        self.run_sparse_sample(
            iterations=200,
            expected=[[0.982629, 2.0], [3.0, 3.982674]],
            optimizer=RectifiedAdam(
                lr=1e-3, total_steps=200, warmup_proportion=0.1, min_lr=1e-5,
            ),
        )

    def test_dense_sample_with_lookahead(self):
        # Expected values are obtained from the original implementation
        # of Ranger
        self.run_dense_sample(
            iterations=100,
            expected=[[0.993126, 1.992901], [2.993283, 3.993261]],
            optimizer=Lookahead(
                RectifiedAdam(lr=1e-3, beta_1=0.95,),
                sync_period=6,
                slow_step_size=0.45,
            ),
        )

    def test_sparse_sample_with_lookahead(self):
        # Expected values are obtained from the previous implementation
        # of Ranger.
        self.run_sparse_sample(
            iterations=150,
            expected=[[0.988156, 2.0], [3.0, 3.988291]],
            optimizer=Lookahead(
                RectifiedAdam(lr=1e-3, beta_1=0.95,),
                sync_period=6,
                slow_step_size=0.45,
            ),
        )

    def test_get_config(self):
        opt = RectifiedAdam(lr=1e-4)
        config = opt.get_config()
        self.assertEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["total_steps"], 0)


def test_serialization():
    optimizer = RectifiedAdam(
        lr=1e-3, total_steps=10000, warmup_proportion=0.1, min_lr=1e-5,
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
