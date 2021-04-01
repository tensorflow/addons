# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for LARS Optimizer."""


import numpy as np
import pytest

import tensorflow as tf

from tensorflow_addons.optimizers import lars
from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_lars_gradient_one_step():
    for dtype in [tf.float32, tf.float64]:
        shape = [3, 3]
        var_np = np.ones(shape)
        grad_np = np.ones(shape)
        lr_np = 0.1
        m_np = 0.9
        wd_np = 0.1
        ep_np = 1e-5
        eeta = 0.1
        vel_np = np.zeros(shape)

        var = tf.Variable(var_np, dtype=dtype)
        grad = tf.Variable(grad_np, dtype=dtype)
        opt = lars.LARS(
            learning_rate=lr_np,
            momentum=m_np,
            weight_decay=wd_np,
            eeta=eeta,
            epsilon=ep_np,
        )

        test_utils.assert_allclose_according_to_type(var_np, var)

        opt.apply_gradients([(grad, var)])

        w_norm = np.linalg.norm(var_np.flatten(), ord=2)
        g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
        trust_ratio = eeta * w_norm / (g_norm + wd_np * w_norm + ep_np)
        scaled_lr = lr_np * trust_ratio
        grad_np = grad_np + wd_np * var_np

        vel_np = m_np * vel_np + scaled_lr * grad_np
        var_np -= vel_np

        test_utils.assert_allclose_according_to_type(var_np, var)
        test_utils.assert_allclose_according_to_type(
            vel_np, opt.get_slot(var, "momentum")
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_lars_gradient_multi_step():
    for dtype in [tf.float32, tf.float64]:
        shape = [3, 3]
        var_np = np.ones(shape)
        grad_np = np.ones(shape)
        lr_np = 0.1
        m_np = 0.9
        wd_np = 0.1
        ep_np = 1e-5
        eeta = 0.1
        vel_np = np.zeros(shape)

        var = tf.Variable(var_np, dtype=dtype)
        grad = tf.Variable(grad_np, dtype=dtype)
        opt = lars.LARS(
            learning_rate=lr_np,
            momentum=m_np,
            eeta=eeta,
            weight_decay=wd_np,
            epsilon=ep_np,
        )

        test_utils.assert_allclose_according_to_type(var_np, var)

        for _ in range(10):
            opt.apply_gradients([(grad, var)])

            w_norm = np.linalg.norm(var_np.flatten(), ord=2)
            g_norm = np.linalg.norm(grad_np.flatten(), ord=2)
            trust_ratio = eeta * w_norm / (g_norm + wd_np * w_norm + ep_np)
            scaled_lr = lr_np * trust_ratio
            grad_np = grad_np + wd_np * var_np

            vel_np = m_np * vel_np + scaled_lr * grad_np
            var_np -= vel_np

            test_utils.assert_allclose_according_to_type(var_np, var)
            test_utils.assert_allclose_according_to_type(
                vel_np, opt.get_slot(var, "momentum")
            )


def test_serialization():
    optimizer = lars.LARS(1e-4)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
