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
"""Tests for LazyAdam."""


import numpy as np
import tensorflow as tf

from tensorflow_addons.optimizers import lazy_adam
from tensorflow_addons.utils import test_utils
import pytest


def adam_update_numpy(
    param, g_t, t, m, v, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7
):
    lr_t = lr * np.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - lr_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
    local_step = tf.cast(opt.iterations + 1, dtype)
    beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse(dtype):
    # TODO: remove the with tf.device when the execution on cpu is enforced
    # See #1682 to track it.
    with tf.device("CPU:0"):
        _test_sparse(dtype)


def _test_sparse(dtype):
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
    opt = lazy_adam.LazyAdam()

    # Fetch params to validate initial values
    np.testing.assert_allclose([1.0, 1.0, 2.0], var0.numpy(), 1e-6, 1e-6)
    np.testing.assert_allclose([3.0, 3.0, 4.0], var1.numpy(), 1e-6, 1e-6)

    # Run 3 steps of Adam
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64])
@pytest.mark.with_device(["cpu", "gpu"])
def test_sparse_device_placement(dtype):

    # If a GPU is available, tests that all optimizer ops can be placed on
    # it (i.e. they have GPU kernels).
    var = tf.Variable([[1.0], [2.0]])
    indices = tf.constant([0, 1], dtype=dtype)

    def g_sum():
        return tf.math.reduce_sum(tf.gather(var, indices))

    optimizer = lazy_adam.LazyAdam(3.0)
    optimizer.minimize(g_sum, var_list=[var])


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_sparse_repeated_indices(dtype):
    # todo: remove the with tf.device once the placement on cpu is enforced.
    with tf.device("CPU:0"):
        repeated_index_update_var = tf.Variable([[1], [2]], dtype=dtype)
        aggregated_update_var = tf.Variable([[1], [2]], dtype=dtype)
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
        repeated_update_opt = lazy_adam.LazyAdam()
        aggregated_update_opt = lazy_adam.LazyAdam()
        for _ in range(3):
            repeated_update_opt.apply_gradients(
                [(grad_repeated_index, repeated_index_update_var)]
            )
            aggregated_update_opt.apply_gradients(
                [(grad_aggregated, aggregated_update_var)]
            )
            np.testing.assert_allclose(
                aggregated_update_var.numpy(), repeated_index_update_var.numpy()
            )


@pytest.mark.parametrize("use_callable_params", [True, False])
@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_basic(use_callable_params, dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)

    def learning_rate():
        return 0.001

    if not use_callable_params:
        learning_rate = learning_rate()

    opt = lazy_adam.LazyAdam(learning_rate=learning_rate)

    # Run 3 steps of Adam
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_tensor_learning_rate(dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)
    opt = lazy_adam.LazyAdam(tf.constant(0.001))

    # Run 3 steps of Adam
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sharing(dtype):
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)
    opt = lazy_adam.LazyAdam()

    # Fetch params to validate initial values
    np.testing.assert_allclose([1.0, 2.0], var0.numpy())
    np.testing.assert_allclose([3.0, 4.0], var1.numpy())

    # Run 3 steps of intertwined Adam1 and Adam2.
    for t in range(3):
        beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
        test_utils.assert_allclose_according_to_type(0.9 ** (t + 1), beta_1_power)
        test_utils.assert_allclose_according_to_type(0.999 ** (t + 1), beta_2_power)
        opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
        var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

        # Validate updated params
        test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def test_slots_unique_eager():
    v1 = tf.Variable(1.0)
    v2 = tf.Variable(1.0)
    opt = lazy_adam.LazyAdam(1.0)
    opt.minimize(lambda: v1 + v2, var_list=[v1, v2])
    # There should be iteration, and two unique slot variables for v1 and v2.
    assert 5 == len(opt.variables())
    assert opt.variables()[0] == opt.iterations


def test_serialization():
    optimizer = lazy_adam.LazyAdam()
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
