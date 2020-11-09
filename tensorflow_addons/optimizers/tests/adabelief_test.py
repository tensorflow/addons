# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for AdaBelief optimizer."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import AdaBelief
from tensorflow_addons.utils import test_utils


def _dtypes_to_test(use_gpu):
    # Based on issue #347 (https://github.com/tensorflow/addons/issues/347)
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel for 'GPU'.
    # So we have to remove tf.half when testing with gpu.
    if use_gpu:
        return [tf.float32, tf.float64]
    else:
        return [tf.half, tf.float32, tf.float64]


def adabelief_update_numpy(
    param,
    g_t,
    t,
    m,
    v,
    lr=0.001,
    weight_decay=0.0,
    rectify=True,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-14,
):
    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * (m_t - g_t) * (m_t - g_t) + epsilon

    m_t_hat = m_t / (1 - beta1 ** (t + 1))
    v_t_hat = np.sqrt(v_t / (1 - beta2 ** (t + 1)))
    if rectify:
        sma_inf = 2.0 / (1.0 - beta2) - 1.0
        sma_t = sma_inf - 2.0 * (t + 1) * (beta2 ** (t + 1)) / (
            1.0 - (beta2 ** (t + 1))
        )

        r_t = (
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )
        sma_threshold = 5.0
        update = np.where(
            sma_t >= sma_threshold, r_t * m_t_hat / (v_t_hat + epsilon), m_t_hat
        )
    else:
        update = m_t_hat / (v_t_hat + epsilon)

    update += weight_decay * param

    param_t = param - lr * update

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

        optimizer = AdaBelief()

        # Fetch params to validate initial values
        np.testing.assert_allclose(np.asanyarray([1.0, 1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 3.0, 4.0]), var1.numpy())

        # Run 3 steps of AdaBelief
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(optimizer, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)

            optimizer.apply_gradients(zip([grads0, grads1], [var0, var1]))
            var0_np, m0, v0 = adabelief_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adabelief_update_numpy(var1_np, grads1_np, t, m1, v1)
            # Validate updated params
            test_utils.assert_allclose_according_to_type(
                var0_np, var0.numpy(), atol=2e-4
            )
            test_utils.assert_allclose_according_to_type(
                var1_np, var1.numpy(), atol=2e-4
            )


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
        epsilon = 1e-14
        decay = 0.5
        weight_decay = 0.01

        opt = AdaBelief(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            decay=decay,
        )

        # Run 3 steps of AdaBelief
        for t in range(3):
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            lr_np = learning_rate / (1 + decay * t)

            var0_np, m0, v0 = adabelief_update_numpy(
                var0_np, grads0_np, t, m0, v0, lr=lr_np, weight_decay=weight_decay
            )
            var1_np, m1, v1 = adabelief_update_numpy(
                var1_np, grads1_np, t, m1, v1, lr=lr_np, weight_decay=weight_decay
            )

            # Validate updated params
            test_utils.assert_allclose_according_to_type(
                var0_np, var0.numpy(), atol=2e-4
            )
            test_utils.assert_allclose_according_to_type(
                var1_np, var1.numpy(), atol=2e-4
            )


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
        epsilon = 1e-14

        opt = AdaBelief(
            learning_rate=lr_schedule, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

        # Run 3 steps of AdaBelief
        for t in range(3):
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            lr_np = learning_rate / (1 + decay * t)

            var0_np, m0, v0 = adabelief_update_numpy(
                var0_np, grads0_np, t, m0, v0, lr=lr_np
            )
            var1_np, m1, v1 = adabelief_update_numpy(
                var1_np, grads1_np, t, m1, v1, lr=lr_np
            )

            # Validate updated params
            test_utils.assert_allclose_according_to_type(
                var0_np, var0.numpy(), atol=2e-4
            )
            test_utils.assert_allclose_according_to_type(
                var1_np, var1.numpy(), atol=2e-4
            )


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
        opt = AdaBelief(lr=tf.constant(0.001))

        # Fetch params to validate initial values
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of AdaBelief
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = adabelief_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adabelief_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(
                var0_np, var0.numpy(), atol=2e-4
            )
            test_utils.assert_allclose_according_to_type(
                var1_np, var1.numpy(), atol=2e-4
            )


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
        opt = AdaBelief()

        # Fetch params to validate initial values
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of AdaBelief
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = adabelief_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adabelief_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def test_minimize_mean_square_loss_with_weight_decay():
    w = tf.Variable([0.1, -0.2, -0.1])
    x = tf.constant([0.4, 0.2, -0.5])

    def loss():
        return tf.reduce_mean(tf.square(x - w))

    opt = AdaBelief(0.02, weight_decay=0.01)

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

        opt = AdaBelief(learning_rate=learning_rate)

        # Run 3 steps of AdaBelief
        for t in range(3):
            beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
            test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = adabelief_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adabelief_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def run_dense_sample(iterations, expected, optimizer):
    var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
    var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

    grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
    grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)
    np.testing.assert_allclose(var_0.read_value(), expected[0], atol=2e-4)
    np.testing.assert_allclose(var_1.read_value(), expected[1], atol=2e-4)


def run_sparse_sample(iterations, expected, optimizer):
    var_0 = tf.Variable([1.0, 2.0])
    var_1 = tf.Variable([3.0, 4.0])

    grad_0 = tf.IndexedSlices(tf.constant([0.1]), tf.constant([0]), tf.constant([2]))
    grad_1 = tf.IndexedSlices(tf.constant([0.04]), tf.constant([1]), tf.constant([2]))

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)
    np.testing.assert_allclose(var_0.read_value(), expected[0], atol=2e-4)
    np.testing.assert_allclose(var_1.read_value(), expected[1], atol=2e-4)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample():
    run_dense_sample(
        iterations=100,
        expected=[[0.9475605, 1.9471607], [2.94784, 3.9478]],
        optimizer=AdaBelief(lr=1e-3),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample():
    run_sparse_sample(
        iterations=200,
        expected=[[0.78374314, 2.0], [3.0, 3.7839816]],
        optimizer=AdaBelief(lr=1e-3),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_amsgrad():
    run_dense_sample(
        iterations=100,
        expected=[[0.9485513, 1.9481515], [2.9488308, 3.9487908]],
        optimizer=AdaBelief(lr=1e-3, amsgrad=True),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_amsgrad():
    run_sparse_sample(
        iterations=200,
        expected=[[0.7947248, 2.0], [3.0, 3.7949643]],
        optimizer=AdaBelief(lr=1e-3, amsgrad=True),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_weight_decay():
    run_dense_sample(
        iterations=100,
        expected=[[0.94657826, 1.9451787], [2.9448593, 3.9438188]],
        optimizer=AdaBelief(lr=1e-3, weight_decay=0.01),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_weight_decay():
    run_sparse_sample(
        iterations=200,
        expected=[[0.7818859, 2.0], [3.0, 3.7761304]],
        optimizer=AdaBelief(lr=1e-3, weight_decay=0.01),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_warmup():
    run_dense_sample(
        iterations=100,
        expected=[[0.9811546, 1.9810544], [2.981224, 3.981214]],
        optimizer=AdaBelief(
            lr=1e-3, total_steps=100, warmup_proportion=0.1, min_lr=1e-5
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_warmup():
    run_sparse_sample(
        iterations=200,
        expected=[[0.9211433, 2.0], [3.0, 3.9211729]],
        optimizer=AdaBelief(
            lr=1e-3, total_steps=200, warmup_proportion=0.1, min_lr=1e-5
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_without_rectify():
    run_dense_sample(
        iterations=100,
        expected=[[0.6672395, 1.6672392], [2.667238, 3.6672378]],
        optimizer=AdaBelief(lr=1e-3, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_without_rectify():
    run_sparse_sample(
        iterations=200,
        expected=[[0.0538935, 2.0], [3.0, 3.0538921]],
        optimizer=AdaBelief(lr=1e-3, rectify=False),
    )


def test_get_config():
    opt = AdaBelief(lr=1e-4)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-4
    assert config["total_steps"] == 0


def test_serialization():
    optimizer = AdaBelief(
        lr=1e-3, total_steps=10000, warmup_proportion=0.1, min_lr=1e-5
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_schedulers():
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 50, 0.5)
    wd_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(2e-3, 25, 0.25)

    run_dense_sample(
        iterations=100,
        expected=[[0.9778532, 1.9773799], [2.977964, 3.977844]],
        optimizer=AdaBelief(learning_rate=lr_scheduler, weight_decay=wd_scheduler),
    )


def test_scheduler_serialization():
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 50, 0.5)
    wd_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(2e-3, 25, 0.25)

    optimizer = AdaBelief(learning_rate=lr_scheduler, weight_decay=wd_scheduler)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()

    assert new_optimizer.get_config()["learning_rate"] == {
        "class_name": "ExponentialDecay",
        "config": lr_scheduler.get_config(),
    }

    assert new_optimizer.get_config()["weight_decay"] == {
        "class_name": "InverseTimeDecay",
        "config": wd_scheduler.get_config(),
    }
