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

import numpy as np
import pytest

import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead


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
    # Expected values are obtained from the previous implementation
    run_dense_sample(
        iterations=100,
        expected=[[0.985769, 1.985269], [2.986119, 3.986068]],
        optimizer=RectifiedAdam(lr=1e-3),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample():
    # Expected values are obtained from the previous implementation
    run_sparse_sample(
        iterations=200,
        expected=[[0.959333, 2.0], [3.0, 3.959632]],
        optimizer=RectifiedAdam(lr=1e-3),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_amsgrad():
    # Expected values are obtained from the official implementation
    # `amsgrad` has no effect because the gradient is fixed
    run_dense_sample(
        iterations=100,
        expected=[[0.985769, 1.985269], [2.986119, 3.986068]],
        optimizer=RectifiedAdam(lr=1e-3, amsgrad=True),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_amsgrad():
    # Expected values are obtained from the official implementation
    # `amsgrad` has no effect because the gradient is fixed
    run_sparse_sample(
        iterations=200,
        expected=[[0.959333, 2.0], [3.0, 3.959632]],
        optimizer=RectifiedAdam(lr=1e-3, amsgrad=True),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_weight_decay():
    # Expected values are obtained from the previous implementation
    run_dense_sample(
        iterations=100,
        expected=[[0.984775, 1.983276], [2.983125, 3.982076]],
        optimizer=RectifiedAdam(lr=1e-3, weight_decay=0.01),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_weight_decay():
    # Expected values are obtained from the previous implementation
    run_sparse_sample(
        iterations=200,
        expected=[[0.957368, 2.0], [3.0, 3.951673]],
        optimizer=RectifiedAdam(lr=1e-3, weight_decay=0.01),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_warmup():
    run_dense_sample(
        iterations=100,
        expected=[[0.994062, 1.993912], [2.994167, 3.994152]],
        optimizer=RectifiedAdam(
            lr=1e-3, total_steps=100, warmup_proportion=0.1, min_lr=1e-5
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_warmup():
    run_sparse_sample(
        iterations=200,
        expected=[[0.982629, 2.0], [3.0, 3.982674]],
        optimizer=RectifiedAdam(
            lr=1e-3, total_steps=200, warmup_proportion=0.1, min_lr=1e-5
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_lookahead():
    # Expected values are obtained from the original implementation
    # of Ranger
    run_dense_sample(
        iterations=100,
        expected=[[0.993126, 1.992901], [2.993283, 3.993261]],
        optimizer=Lookahead(
            RectifiedAdam(lr=1e-3, beta_1=0.95), sync_period=6, slow_step_size=0.45
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_lookahead():
    # Expected values are obtained from the previous implementation
    # of Ranger.
    run_sparse_sample(
        iterations=150,
        expected=[[0.988156, 2.0], [3.0, 3.988291]],
        optimizer=Lookahead(
            RectifiedAdam(lr=1e-3, beta_1=0.95), sync_period=6, slow_step_size=0.45
        ),
    )


def test_get_config():
    opt = RectifiedAdam(lr=1e-4)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-4
    assert config["total_steps"] == 0


def test_serialization():
    optimizer = RectifiedAdam(
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
        expected=[[0.993192, 1.992625], [2.993369, 3.993239]],
        optimizer=RectifiedAdam(learning_rate=lr_scheduler, weight_decay=wd_scheduler),
    )


def test_scheduler_serialization():
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 50, 0.5)
    wd_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(2e-3, 25, 0.25)

    optimizer = RectifiedAdam(learning_rate=lr_scheduler, weight_decay=wd_scheduler)
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


def test_checkpoint_serialization(tmpdir):
    optimizer = RectifiedAdam()
    optimizer2 = RectifiedAdam()

    var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
    var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

    grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
    grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    optimizer.apply_gradients(grads_and_vars)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer)
    checkpoint2 = tf.train.Checkpoint(optimizer=optimizer2)
    model_path = str(tmpdir / "rectified_adam_chkpt")
    checkpoint.write(model_path)
    checkpoint2.read(model_path)

    optimizer2.apply_gradients(grads_and_vars)
