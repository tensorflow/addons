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

import importlib
import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import weight_decay_optimizers

WEIGHT_DECAY = 0.01


def do_test(
    dtype,
    optimizer,
    update_fn,
    do_sparse=False,
    do_decay_var_list=False,
    **optimizer_kwargs,
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
    if do_sparse and test_utils.is_gpu_available():
        pytest.skip("Wait #347 to be fixed")

    # Initialize variables for numpy implementation.
    np_slot_vars0, np_slot_vars1 = {}, {}
    var0_np = np.array([1.0, 2.0], dtype=dtype[0].as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype[0].as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype[0].as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype[0].as_numpy_dtype)
    # Create Tensorflow variables.
    var0 = tf.Variable(var0_np, name="var0_%d" % dtype[1])
    var1 = tf.Variable(var1_np, name="var1_%d" % dtype[1])
    if do_sparse:
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = tf.IndexedSlices(
            tf.constant(grads0_np), tf.constant(grads0_np_indices), tf.constant([2])
        )
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = tf.IndexedSlices(
            tf.constant(grads1_np), tf.constant(grads1_np_indices), tf.constant([2])
        )
    else:
        grads0 = tf.constant(grads0_np)
        grads1 = tf.constant(grads1_np)
    opt = optimizer(**optimizer_kwargs)
    # Create the update op.
    # Run 3 steps of the optimizer
    optimizer_kwargs.pop("exclude_from_weight_decay", None)
    for _ in range(3):
        if do_decay_var_list:
            opt.apply_gradients(
                zip([grads0, grads1], [var0, var1]), decay_var_list=[var0, var1]
            )
        else:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        var0_np, np_slot_vars0 = update_fn(
            var0_np, grads0_np, np_slot_vars0, **optimizer_kwargs
        )
        var1_np, np_slot_vars1 = update_fn(
            var1_np, grads1_np, np_slot_vars1, **optimizer_kwargs
        )
        # Validate updated params
        test_utils.assert_allclose_according_to_type(var0_np, var0.numpy())
        test_utils.assert_allclose_according_to_type(var1_np, var1.numpy())


def do_test_sparse_repeated_indices(dtype, optimizer, **optimizer_kwargs):
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
    if test_utils.is_gpu_available():
        pytest.skip("Wait #347 to be fixed")

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
    _ = opt_repeated.apply_gradients([(grad_repeated_index, repeated_index_update_var)])
    opt_aggregated = optimizer(**optimizer_kwargs)
    _ = opt_aggregated.apply_gradients([(grad_aggregated, aggregated_update_var)])
    np.testing.assert_allclose(
        aggregated_update_var.numpy(), repeated_index_update_var.numpy()
    )
    for _ in range(3):
        opt_repeated.apply_gradients([(grad_repeated_index, repeated_index_update_var)])
        opt_aggregated.apply_gradients([(grad_aggregated, aggregated_update_var)])
        np.testing.assert_allclose(
            aggregated_update_var.numpy(), repeated_index_update_var.numpy()
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
    lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
    slot_vars["m"] = beta1 * slot_vars.get("m", 0) + (1 - beta1) * grad_t
    slot_vars["v"] = beta2 * slot_vars.get("v", 0) + (1 - beta2) * grad_t**2
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


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_sparse_adamw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.AdamW,
        adamw_update_numpy,
        do_sparse=True,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_sparse_repeated_indices_adamw(dtype):
    do_test_sparse_repeated_indices(
        dtype,
        weight_decay_optimizers.AdamW,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_basic_adamw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.AdamW,
        adamw_update_numpy,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_basic_callable_params_adamw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.AdamW,
        adamw_update_numpy,
        learning_rate=lambda: 0.001,
        beta_1=lambda: 0.9,
        beta_2=lambda: 0.999,
        epsilon=1e-8,
        weight_decay=lambda: WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_basic_decay_var_list_adamw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.AdamW,
        adamw_update_numpy,
        do_decay_var_list=True,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


def test_exclude_weight_decay_adamw():
    optimizer = weight_decay_optimizers.AdamW(
        learning_rate=1e-4, weight_decay=1e-4, exclude_from_weight_decay=["var1"]
    )
    var0 = tf.Variable([], name="var0")
    var1 = tf.Variable([], name="var1")
    var1_weight = tf.Variable([], name="var1_weight")

    optimizer._set_decay_var_list([var0, var1, var1_weight])
    assert optimizer._do_use_weight_decay(var0)
    assert not optimizer._do_use_weight_decay(var1)
    assert not optimizer._do_use_weight_decay(var1_weight)


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_var_list_with_exclude_list_adamw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.AdamW,
        adamw_update_numpy,
        do_decay_var_list=True,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=WEIGHT_DECAY,
        exclude_from_weight_decay=["var0_*", "var1_*"],
    )


def test_keras_fit():
    """Check if calling model.fit works."""
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = weight_decay_optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    x, y = np.random.uniform(size=(2, 4, 1))
    model.fit(x, y, epochs=1)


def test_keras_fit_with_schedule():
    """Check if calling model.fit works with wd schedule."""
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    wd_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        WEIGHT_DECAY, decay_steps=10, decay_rate=0.9
    )
    optimizer = weight_decay_optimizers.AdamW(
        learning_rate=1e-4, weight_decay=wd_schedule
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    x, y = np.random.uniform(size=(2, 4, 1))
    model.fit(x, y, epochs=1)


@pytest.mark.with_device(["cpu", "gpu"])
def test_weight_decay_with_piecewise_constant_decay_schedule():
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(2)])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    wd_schedule = tf.optimizers.schedules.PiecewiseConstantDecay([2], [1e-4, 1e-5])
    optimizer = weight_decay_optimizers.SGDW(
        learning_rate=1e-2, weight_decay=wd_schedule
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    x, y = np.random.uniform(size=(2, 4, 1))
    model.fit(x, y, batch_size=1, epochs=1)


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_sparse_sgdw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.SGDW,
        sgdw_update_numpy,
        do_sparse=True,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_sparse_repeated_indices_sgdw(dtype):
    do_test_sparse_repeated_indices(
        dtype,
        weight_decay_optimizers.SGDW,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_basic_sgdw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.SGDW,
        sgdw_update_numpy,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_basic_callable_params_sgdw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.SGDW,
        sgdw_update_numpy,
        learning_rate=lambda: 0.001,
        momentum=lambda: 0.9,
        weight_decay=lambda: WEIGHT_DECAY,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_basic_decay_var_list_sgdw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.SGDW,
        sgdw_update_numpy,
        do_decay_var_list=True,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )


def test_exclude_weight_decay_sgdw():
    optimizer = weight_decay_optimizers.SGDW(
        learning_rate=0.01, weight_decay=1e-4, exclude_from_weight_decay=["var1"]
    )
    var0 = tf.Variable([], name="var0")
    var1 = tf.Variable([], name="var1")
    var1_weight = tf.Variable([], name="var1_weight")

    optimizer._set_decay_var_list([var0, var1, var1_weight])
    assert optimizer._do_use_weight_decay(var0)
    assert not optimizer._do_use_weight_decay(var1)
    assert not optimizer._do_use_weight_decay(var1_weight)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_var_list_with_exclude_list_sgdw(dtype):
    do_test(
        dtype,
        weight_decay_optimizers.SGDW,
        sgdw_update_numpy,
        do_decay_var_list=True,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        exclude_from_weight_decay=["var0_*", "var1_*"],
    )


if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
    optimizer_class = tf.keras.optimizers.legacy.SGD
else:
    optimizer_class = tf.keras.optimizers.SGD


@pytest.mark.parametrize(
    "optimizer",
    [
        weight_decay_optimizers.SGDW,
        weight_decay_optimizers.extend_with_decoupled_weight_decay(optimizer_class),
    ],
)
@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
def test_optimizer_basic(dtype, optimizer):
    do_test(
        dtype,
        optimizer,
        sgdw_update_numpy,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        weight_decay_optimizers.SGDW,
        weight_decay_optimizers.extend_with_decoupled_weight_decay(optimizer_class),
    ],
)
@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_optimizer_sparse(dtype, optimizer):
    do_test_sparse_repeated_indices(
        dtype, optimizer, learning_rate=0.001, momentum=0.9, weight_decay=WEIGHT_DECAY
    )


def test_serialization():
    optimizer = weight_decay_optimizers.AdamW(
        learning_rate=1e-4, weight_decay=1e-4, exclude_from_weight_decay=["var1"]
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()


def test_serialization_with_wd_schedule():
    wd_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        WEIGHT_DECAY, decay_steps=10, decay_rate=0.9
    )
    optimizer = weight_decay_optimizers.AdamW(
        learning_rate=1e-4, weight_decay=wd_schedule
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
