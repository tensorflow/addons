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
"""Tests for Cyclical Learning Rate."""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_addons.utils import test_utils
import numpy as np

from tensorflow_addons.optimizers import cyclical_learning_rate


def _maybe_serialized(lr_decay, serialize_and_deserialize):
    if serialize_and_deserialize:
        serialized = tf.keras.optimizers.learning_rate_schedule.serialize(lr_decay)
        return tf.keras.optimizers.learning_rate_schedule.deserialize(serialized)
    else:
        return lr_decay


@test_utils.run_all_in_graph_and_eager_modes
@parameterized.named_parameters(("NotSerialized", False), ("Serialized", True))
class CyclicalLearningRateTest(tf.test.TestCase, parameterized.TestCase):
    def testTriangularCyclicalLearningRate(self, serialize):
        self.skipTest("Failing. See https://github.com/tensorflow/addons/issues/1203")
        initial_learning_rate = 0.1
        maximal_learning_rate = 1
        step_size = 4000
        step = tf.resource_variable_ops.ResourceVariable(0)
        triangular_cyclical_lr = cyclical_learning_rate.TriangularCyclicalLearningRate(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
        )
        triangular_cyclical_lr = _maybe_serialized(triangular_cyclical_lr, serialize)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        expected = np.concatenate(
            [
                np.linspace(initial_learning_rate, maximal_learning_rate, num=2001)[1:],
                np.linspace(maximal_learning_rate, initial_learning_rate, num=2001)[1:],
            ]
        )

        for expected_value in expected:
            self.assertAllClose(
                self.evaluate(triangular_cyclical_lr(step)), expected_value, 1e-6
            )
            self.evaluate(step.assign_add(1))

    def testTriangular2CyclicalLearningRate(self, serialize):
        self.skipTest("Failing. See https://github.com/tensorflow/addons/issues/1203")
        initial_learning_rate = 0.1
        maximal_learning_rate = 1
        step_size = 4000
        step = tf.resource_variable_ops.ResourceVariable(0)
        triangular2_cyclical_lr = cyclical_learning_rate.Triangular2CyclicalLearningRate(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
        )
        triangular2_cyclical_lr = _maybe_serialized(triangular2_cyclical_lr, serialize)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        middle_learning_rate = (maximal_learning_rate + initial_learning_rate) / 2
        expected = np.concatenate(
            [
                np.linspace(initial_learning_rate, maximal_learning_rate, num=2001)[1:],
                np.linspace(maximal_learning_rate, initial_learning_rate, num=2001)[1:],
                np.linspace(initial_learning_rate, middle_learning_rate, num=2001)[1:],
                np.linspace(middle_learning_rate, initial_learning_rate, num=2001)[1:],
            ]
        )

        for expected_value in expected:
            self.assertAllClose(
                self.evaluate(triangular2_cyclical_lr(step)), expected_value, 1e-6
            )
            self.evaluate(step.assign_add(1))

    def testExponentialCyclicalLearningRate(self, serialize):
        self.skipTest("Failing. See https://github.com/tensorflow/addons/issues/1203")
        initial_learning_rate = 0.1
        maximal_learning_rate = 1
        step_size = 4000
        gamma = 0.996

        step = tf.resource_variable_ops.ResourceVariable(0)
        exponential_cyclical_lr = cyclical_learning_rate.ExponentialCyclicalLearningRate(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
            gamma=gamma,
        )
        exponential_cyclical_lr = _maybe_serialized(exponential_cyclical_lr, serialize)

        self.evaluate(tf.compat.v1.global_variables_initializer())

        for i in range(1, 8001):
            non_bounded_value = np.abs(
                i / 2000.0 - 2 * np.floor(1 + i / (2 * 2000)) + 1
            )
            expected = initial_learning_rate + (
                maximal_learning_rate - initial_learning_rate
            ) * np.maximum(0, (1 - non_bounded_value)) * (gamma ** i)
            self.assertAllClose(
                self.evaluate(exponential_cyclical_lr(step)), expected, 1e-6
            )
            self.evaluate(step.assign_add(1))

    def testCustomCyclicalLearningRate(self, serialize):
        self.skipTest("Failing. See https://github.com/tensorflow/addons/issues/1203")
        initial_learning_rate = 0.1
        maximal_learning_rate = 1
        step_size = 4000

        def scale_fn(x):
            return 1 / (5 ** (x * 0.0001))

        step = tf.resource_variable_ops.ResourceVariable(0)
        custom_cyclical_lr = cyclical_learning_rate.CyclicalLearningRate(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
            scale_fn=scale_fn,
        )
        custom_cyclical_lr = _maybe_serialized(custom_cyclical_lr, serialize)

        self.evaluate(tf.compat.v1.global_variables_initializer())

        for i in range(1, 8001):
            non_bounded_value = np.abs(
                i / 2000.0 - 2 * np.floor(1 + i / (2 * 2000)) + 1
            )
            expected = initial_learning_rate + (
                maximal_learning_rate - initial_learning_rate
            ) * np.maximum(0, 1 - non_bounded_value) * scale_fn(i)
            self.assertAllClose(self.evaluate(custom_cyclical_lr(step)), expected, 1e-6)
            self.evaluate(step.assign_add(1))
