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
"""Tests for Yogi optimizer."""


import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import yogi
from tensorflow_addons.utils import test_utils


def yogi_update_numpy(
    param,
    g_t,
    t,
    m,
    v,
    alpha=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-3,
    l1reg=0.0,
    l2reg=0.0,
):
    """Performs Yogi parameter update using numpy.

    Args:
      param: An numpy ndarray of the current parameter.
      g_t: An numpy ndarray of the current gradients.
      t: An numpy ndarray of the current time step.
      m: An numpy ndarray of the 1st moment estimates.
      v: An numpy ndarray of the 2nd moment estimates.
      alpha: A float value of the learning rate.
      beta1: A float value of the exponential decay rate for the 1st moment
        estimates.
      beta2: A float value of the exponential decay rate for the 2nd moment
         estimates.
      epsilon: A float of a small constant for numerical stability.
      l1reg: A float value of L1 regularization
      l2reg: A float value of L2 regularization
    Returns:
      A tuple of numpy ndarrays (param_t, m_t, v_t) representing the
      updated parameters for `param`, `m`, and `v` respectively.
    """
    beta1 = np.array(beta1, dtype=param.dtype)
    beta2 = np.array(beta2, dtype=param.dtype)

    alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

    m_t = beta1 * m + (1 - beta1) * g_t
    g2_t = g_t * g_t
    v_t = v - (1 - beta2) * np.sign(v - g2_t) * g2_t

    per_coord_lr = alpha_t / (np.sqrt(v_t) + epsilon)
    param_t = param - per_coord_lr * m_t

    if l1reg > 0:
        param_t = (param_t - l1reg * per_coord_lr * np.sign(param_t)) / (
            1 + l2reg * per_coord_lr
        )
        print(param_t.dtype)
        param_t[np.abs(param_t) < l1reg * per_coord_lr] = 0.0
    elif l2reg > 0:
        param_t = param_t / (1 + l2reg * per_coord_lr)
    return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
    local_step = tf.cast(opt.iterations + 1, dtype)
    beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
    beta_1_power = tf.math.pow(beta_1_t, local_step)
    beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
    beta_2_power = tf.math.pow(beta_2_t, local_step)
    return (beta_1_power, beta_2_power)


def _dtypes_to_test(use_gpu):
    if use_gpu:
        return [tf.dtypes.float32, tf.dtypes.float64]
    else:
        return [tf.dtypes.half, tf.dtypes.float32, tf.dtypes.float64]


def do_test_sparse(beta1=0.0, l1reg=0.0, l2reg=0.0):
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np)
        var1 = tf.Variable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = tf.IndexedSlices(
            tf.constant(grads0_np), tf.constant(grads0_np_indices), tf.constant([2])
        )
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = tf.IndexedSlices(
            tf.constant(grads1_np), tf.constant(grads1_np_indices), tf.constant([2])
        )
        opt = yogi.Yogi(
            beta1=beta1,
            l1_regularization_strength=l1reg,
            l2_regularization_strength=l2reg,
            initial_accumulator_value=1.0,
        )

        # Fetch params to validate initial values.
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of Yogi.
        for t in range(1, 4):
            beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(beta1**t, beta1_power)
            test_utils.assert_allclose_according_to_type(0.999**t, beta2_power)
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = yogi_update_numpy(
                var0_np, grads0_np, t, m0, v0, beta1=beta1, l1reg=l1reg, l2reg=l2reg
            )
            var1_np, m1, v1 = yogi_update_numpy(
                var1_np, grads1_np, t, m1, v1, beta1=beta1, l1reg=l1reg, l2reg=l2reg
            )

            # Validate updated params.
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse():
    do_test_sparse()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_regularization():
    do_test_sparse(l1reg=0.1, l2reg=0.2)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_momentum():
    do_test_sparse(beta1=0.9)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_momentum_regularization():
    do_test_sparse(beta1=0.9, l1reg=0.1, l2reg=0.2)


def test_sparse_repeated_indices():
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
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
        opt1 = yogi.Yogi()
        opt2 = yogi.Yogi()

        np.testing.assert_allclose(
            aggregated_update_var.numpy(),
            repeated_index_update_var.numpy(),
        )

        for _ in range(3):
            opt1.apply_gradients([(grad_repeated_index, repeated_index_update_var)])
            opt2.apply_gradients([(grad_aggregated, aggregated_update_var)])

        np.testing.assert_allclose(
            aggregated_update_var.numpy(),
            repeated_index_update_var.numpy(),
        )


def do_test_basic(beta1=0.0, l1reg=0.0, l2reg=0.0):
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np)
        var1 = tf.Variable(var1_np)
        grads0 = tf.constant(grads0_np)
        grads1 = tf.constant(grads1_np)

        opt = yogi.Yogi(
            beta1=beta1,
            l1_regularization_strength=l1reg,
            l2_regularization_strength=l2reg,
            initial_accumulator_value=1.0,
        )

        # Fetch params to validate initial values.
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of Yogi.
        for t in range(1, 4):
            beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(beta1**t, beta1_power)
            test_utils.assert_allclose_according_to_type(0.999**t, beta2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = yogi_update_numpy(
                var0_np, grads0_np, t, m0, v0, beta1=beta1, l1reg=l1reg, l2reg=l2reg
            )
            var1_np, m1, v1 = yogi_update_numpy(
                var1_np, grads1_np, t, m1, v1, beta1=beta1, l1reg=l1reg, l2reg=l2reg
            )

            # Validate updated params.
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_basic():
    do_test_basic()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_basic_regularization():
    do_test_basic(l1reg=0.1, l2reg=0.2)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_basic_momentum():
    do_test_basic(beta1=0.9)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_basic_momentum_regularization():
    do_test_basic(beta1=0.9, l1reg=0.1, l2reg=0.2)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_tensor_learning_rate():
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np)
        var1 = tf.Variable(var1_np)
        grads0 = tf.constant(grads0_np)
        grads1 = tf.constant(grads1_np)
        opt = yogi.Yogi(tf.constant(0.01), initial_accumulator_value=1.0)

        # Fetch params to validate initial values.
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of Yogi.
        for t in range(1, 4):
            beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9**t, beta1_power)
            test_utils.assert_allclose_according_to_type(0.999**t, beta2_power)

            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

            var0_np, m0, v0 = yogi_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = yogi_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params.
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sharing():
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 1.0, 0.0, 1.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = tf.Variable(var0_np)
        var1 = tf.Variable(var1_np)
        grads0 = tf.constant(grads0_np)
        grads1 = tf.constant(grads1_np)
        opt = yogi.Yogi(initial_accumulator_value=1.0)

        # Fetch params to validate initial values.
        np.testing.assert_allclose(np.asanyarray([1.0, 2.0]), var0.numpy())
        np.testing.assert_allclose(np.asanyarray([3.0, 4.0]), var1.numpy())

        # Run 3 steps of intertwined Yogi1 and Yogi2.
        for t in range(1, 4):
            beta1_power, beta2_power = get_beta_accumulators(opt, dtype)
            test_utils.assert_allclose_according_to_type(0.9**t, beta1_power)
            test_utils.assert_allclose_according_to_type(0.999**t, beta2_power)
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
            var0_np, m0, v0 = yogi_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = yogi_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params.
            test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
            test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def test_get_config():
    opt = yogi.Yogi(1e-4)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-4


def test_serialization():
    optimizer = yogi.Yogi(1e-4)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
