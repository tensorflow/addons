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
"""Tests for AdaBelief optimizer."""

import numpy as np
import pytest

import tensorflow.compat.v2 as tf
from tensorflow_addons.optimizers import AdaBelief, Lookahead


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
        expected=[[0.66723955, 1.6672393], [2.6672382, 3.6672382]],
        optimizer=AdaBelief(lr=1e-3, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample():
    # Expected values are obtained from the previous implementation
    run_sparse_sample(
        iterations=200,
        expected=[[0.0538936, 2.0], [3.0, 3.0538926]],
        optimizer=AdaBelief(lr=1e-3, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_amsgrad():
    # Expected values are obtained from the official implementation
    # `amsgrad` has no effect because the gradient is fixed
    run_dense_sample(
        iterations=100,
        expected=[[0.67249274, 1.6724932], [2.6724923, 3.6724923]],
        optimizer=AdaBelief(lr=1e-3, amsgrad=True, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_amsgrad():
    # Expected values are obtained from the official implementation
    # `amsgrad` has no effect because the gradient is fixed
    run_sparse_sample(
        iterations=200,
        expected=[[0.09575394, 2.0], [3.0, 3.0957537]],
        optimizer=AdaBelief(lr=1e-3, amsgrad=True, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_weight_decay():
    # Expected values are obtained from the previous implementation
    run_dense_sample(
        iterations=100,
        expected=[[0.66637343, 1.6653734], [2.6643748, 3.6633751]],
        optimizer=AdaBelief(lr=1e-3, weight_decay=0.01, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_weight_decay():
    # Expected values are obtained from the previous implementation
    run_sparse_sample(
        iterations=200,
        expected=[[0.05264655, 2.0], [3.0, 3.0466535]],
        optimizer=AdaBelief(lr=1e-3, weight_decay=0.01, rectify=False),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_warmup():
    run_dense_sample(
        iterations=100,
        expected=[[0.85635465, 1.8563547], [2.8563545, 3.8563545]],
        optimizer=AdaBelief(
            lr=1e-3, total_steps=100, warmup_proportion=0.1, min_lr=1e-5, rectify=False
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_warmup():
    # Expected values are obtained from the previous implementation
    run_sparse_sample(
        iterations=200,
        expected=[[0.8502214, 2.0], [3.0, 3.85022]],
        optimizer=AdaBelief(
            lr=1e-3, total_steps=100, warmup_proportion=0.1, min_lr=1e-5, rectify=False
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_rectify():
    run_sparse_sample(
        iterations=200,
        expected=[[0.7836679, 2.0], [3.0, 3.7839665]],
        optimizer=AdaBelief(lr=1e-3, rectify=True),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_rectify():
    run_sparse_sample(
        iterations=200,
        expected=[[0.7836679, 2.0], [3.0, 3.7839665]],
        optimizer=AdaBelief(lr=1e-3, rectify=True),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_lookahead():
    # Expected values are obtained from the original implementation
    # of Ranger
    run_dense_sample(
        iterations=100,
        expected=[[0.88910455, 1.889104], [2.8891046, 3.8891046]],
        optimizer=Lookahead(
            AdaBelief(lr=1e-3, beta_1=0.95, rectify=False),
            sync_period=6,
            slow_step_size=0.45,
        ),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_lookahead():
    # Expected values are obtained from the previous implementation
    # of Ranger.
    run_sparse_sample(
        iterations=150,
        expected=[[0.8114481, 2.0], [3.0, 3.8114486]],
        optimizer=Lookahead(
            AdaBelief(lr=1e-3, beta_1=0.95, rectify=False),
            sync_period=6,
            slow_step_size=0.45,
        ),
    )


def test_get_config():
    opt = AdaBelief(lr=1e-4)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-4
    assert config["total_steps"] == 0


def test_serialization():
    optimizer = AdaBelief(
        lr=1e-3, total_steps=10000, warmup_proportion=0.1, min_lr=1e-5, rectify=False
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
        expected=[[0.84216374, 1.8420818], [2.8420012, 3.841918]],
        optimizer=AdaBelief(
            learning_rate=lr_scheduler, weight_decay=wd_scheduler, rectify=False
        ),
    )


def test_scheduler_serialization():
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 50, 0.5)
    wd_scheduler = tf.keras.optimizers.schedules.InverseTimeDecay(2e-3, 25, 0.25)

    optimizer = AdaBelief(
        learning_rate=lr_scheduler, weight_decay=wd_scheduler, rectify=False
    )
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
    optimizer = AdaBelief()
    optimizer2 = AdaBelief()

    var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
    var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

    grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
    grad_1 = tf.constant([0.03, 0.04], dtype=tf.dtypes.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    optimizer.apply_gradients(grads_and_vars)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer)
    checkpoint2 = tf.train.Checkpoint(optimizer=optimizer2)
    model_path = str(tmpdir / "adabelief_chkpt")
    checkpoint.write(model_path)
    checkpoint2.read(model_path)

    optimizer2.apply_gradients(grads_and_vars)
