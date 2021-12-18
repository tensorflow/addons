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
import pytest

import tensorflow as tf

from tensorflow_addons.optimizers import lamb
from tensorflow_addons.utils import test_utils


def _dtypes_to_test(use_gpu):
    # Based on issue #347 (https://github.com/tensorflow/addons/issues/347)
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel for 'GPU'.
    # So we have to remove tf.half when testing with gpu.
    if use_gpu:
        return [tf.float32, tf.float64]
    else:
        return [tf.half, tf.float32, tf.float64]


def lamb_update_numpy(
    param, g_t, t, m, v, lr=0.001, lamb_wd=0.0, beta1=0.9, beta2=0.999, epsilon=1e-6
):

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    m_t_hat = m_t / (1 - beta1 ** (t + 1))
    v_t_hat = v_t / (1 - beta2 ** (t + 1))
    update = m_t_hat / (np.sqrt(v_t_hat) + epsilon)

    update += lamb_wd * param

    w_norm = linalg.norm(param, ord=2)
    g_norm = linalg.norm(update, ord=2)
    ratio = np.where(w_norm > 0, np.where(g_norm > 0, (w_norm / g_norm), 1.0), 1.0)

    param_t = param - ratio * lr * update
    return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
    local_step = tf.cast(opt.iterations + 1, dtype)
    beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse():
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
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
            tf.constant(grads0_np_indices),
            tf.constant([3]),
        )
        grads1_np_indices = np.array([0, 2], dtype=np.int32)
        grads1 = tf.IndexedSlices(
            tf.constant(grads1_np[grads1_np_indices]),
            tf.constant(grads1_np_indices),
            tf.constant([3]),
        )
        opt = lamb.LAMB()

        # Fetch params to validate initial values
        np.testing.assert_allclose(np.asanyarray([1.0, 1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 3.0, 4.0]), var1.numpy())

        # Run 3 steps of LAMB
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_basic_with_learning_rate_decay():
    for i, dtype in enumerate(_dtypes_to_test(use_gpu=test_utils.is_gpu_available())):
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
            weight_decay=lamb_wd,
            decay=decay,
        )

        # Run 3 steps of LAMB
        for t in range(3):
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            lr_np = learning_rate / (1 + decay * t)

            var0_np, m0, v0 = lamb_update_numpy(
                var0_np, grads0_np, t, m0, v0, lr=lr_np, lamb_wd=lamb_wd
            )
            var1_np, m1, v1 = lamb_update_numpy(
                var1_np, grads1_np, t, m1, v1, lr=lr_np, lamb_wd=lamb_wd
            )

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_basic_with_learning_rate_inverse_time_decay():
    for i, dtype in enumerate(_dtypes_to_test(use_gpu=test_utils.is_gpu_available())):
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
            learning_rate, decay_steps=1.0, decay_rate=decay
        )
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-7

        opt = lamb.LAMB(
            learning_rate=lr_schedule, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

        # Run 3 steps of LAMB
        for t in range(3):
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            lr_np = learning_rate / (1 + decay * t)

            var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0, v0, lr=lr_np)
            var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1, v1, lr=lr_np)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_tensor_learning_rate():
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
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

        # Fetch params to validate initial values
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of LAMB
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sharing():
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
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

        # Fetch params to validate initial values
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of intertwined LAMB1 and LAMB2.
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def test_minimize_mean_square_loss_with_weight_decay():
    w = tf.Variable([0.1, -0.2, -0.1])
    x = tf.constant([0.4, 0.2, -0.5])

    def loss():
        return tf.reduce_mean(tf.square(x - w))

    opt = lamb.LAMB(0.02, weight_decay=0.01)

    # Run 200 steps
    for _ in range(200):
        opt.minimize(loss, [w])
    # Validate updated params
    np.testing.assert_allclose(
        w.numpy(), np.asanyarray([0.4, 0.2, -0.5]), rtol=1e-2, atol=1e-2
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_resource():
    for i, dtype in enumerate(_dtypes_to_test(use_gpu=test_utils.is_gpu_available())):
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

        def learning_rate():
            return 0.001

        opt = lamb.LAMB(learning_rate=learning_rate)

        # Run 3 steps of LAMB
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def test_get_config():
    opt = lamb.LAMB(1e-4)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-4


def test_exclude_weight_decay():
    opt = lamb.LAMB(0.01, weight_decay=0.01, exclude_from_weight_decay=["var1"])
    assert opt._do_use_weight_decay(tf.Variable([], name="var0"))
    assert not opt._do_use_weight_decay(tf.Variable([], name="var1"))
    assert not opt._do_use_weight_decay(tf.Variable([], name="var1_weight"))


def test_exclude_layer_adaptation():
    opt = lamb.LAMB(0.01, exclude_from_layer_adaptation=["var1"])
    assert opt._do_layer_adaptation(tf.Variable([], name="var0"))
    assert not opt._do_layer_adaptation(tf.Variable([], name="var1"))
    assert not opt._do_layer_adaptation(tf.Variable([], name="var1_weight"))


def test_serialization():
    optimizer = lamb.LAMB(
        1e-4, weight_decay_rate=0.01, exclude_from_weight_decay=["var1"]
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


def test_weight_decay_rate_deprecation():
    with pytest.deprecated_call():
        opt = lamb.LAMB(0.01, weight_decay_rate=0.01)
        config = opt.get_config()
        assert config["weight_decay"] == 0.01
