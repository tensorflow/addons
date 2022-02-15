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

import pytest
import numpy as np

from tensorflow_addons.optimizers import cyclical_learning_rate


def _maybe_serialized(lr_decay, serialize_and_deserialize):
    if serialize_and_deserialize:
        serialized = lr_decay.get_config()
        return lr_decay.from_config(serialized)
    else:
        return lr_decay


@pytest.mark.parametrize("serialize", [True, False])
def test_triangular_cyclical_learning_rate(serialize):
    initial_learning_rate = 0.1
    max_learning_rate = 1
    step_size = 40
    triangular_cyclical_lr = cyclical_learning_rate.TriangularCyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=max_learning_rate,
        step_size=step_size,
    )
    triangular_cyclical_lr = _maybe_serialized(triangular_cyclical_lr, serialize)

    expected = np.concatenate(
        [
            np.linspace(initial_learning_rate, max_learning_rate, num=step_size + 1),
            np.linspace(max_learning_rate, initial_learning_rate, num=step_size + 1)[
                1:
            ],
        ]
    )

    for step, expected_value in enumerate(expected):
        np.testing.assert_allclose(triangular_cyclical_lr(step), expected_value, 1e-6)


@pytest.mark.parametrize("serialize", [True, False])
def test_triangular2_cyclical_learning_rate(serialize):
    initial_lr = 0.1
    maximal_lr = 1
    step_size = 30
    triangular2_lr = cyclical_learning_rate.Triangular2CyclicalLearningRate(
        initial_learning_rate=initial_lr,
        maximal_learning_rate=maximal_lr,
        step_size=step_size,
    )
    triangular2_lr = _maybe_serialized(triangular2_lr, serialize)

    middle_lr = (maximal_lr + initial_lr) / 2
    expected = np.concatenate(
        [
            np.linspace(initial_lr, maximal_lr, num=step_size + 1),
            np.linspace(maximal_lr, initial_lr, num=step_size + 1)[1:],
            np.linspace(initial_lr, middle_lr, num=step_size + 1)[1:],
            np.linspace(middle_lr, initial_lr, num=step_size + 1)[1:],
        ]
    )

    for step, expected_value in enumerate(expected):
        np.testing.assert_allclose(triangular2_lr(step).numpy(), expected_value, 1e-6)


@pytest.mark.parametrize("serialize", [True, False])
def test_exponential_cyclical_learning_rate(serialize):
    initial_learning_rate = 0.1
    maximal_learning_rate = 1
    step_size = 2000
    gamma = 0.996

    step = 0
    exponential_cyclical_lr = cyclical_learning_rate.ExponentialCyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size,
        gamma=gamma,
    )
    exponential_cyclical_lr = _maybe_serialized(exponential_cyclical_lr, serialize)

    for i in range(0, 8001):
        non_bounded_value = np.abs(i / 2000.0 - 2 * np.floor(1 + i / (2 * 2000)) + 1)
        expected = initial_learning_rate + (
            maximal_learning_rate - initial_learning_rate
        ) * np.maximum(0, (1 - non_bounded_value)) * (gamma**i)
        computed = exponential_cyclical_lr(step).numpy()
        np.testing.assert_allclose(computed, expected, 1e-6)
        step += 1


@pytest.mark.parametrize("serialize", [True, False])
def test_custom_cyclical_learning_rate(serialize):
    initial_learning_rate = 0.1
    maximal_learning_rate = 1
    step_size = 4000

    def scale_fn(x):
        return 1 / (5 ** (x * 0.0001))

    custom_cyclical_lr = cyclical_learning_rate.CyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size,
        scale_fn=scale_fn,
    )
    custom_cyclical_lr = _maybe_serialized(custom_cyclical_lr, serialize)

    for step in range(1, 8001):
        cycle = np.floor(1 + step / (2 * step_size))
        non_bounded_value = np.abs(step / step_size - 2 * cycle + 1)
        expected = initial_learning_rate + (
            maximal_learning_rate - initial_learning_rate
        ) * np.maximum(0, 1 - non_bounded_value) * scale_fn(cycle)
        np.testing.assert_allclose(
            custom_cyclical_lr(step).numpy(), expected, 1e-6, 1e-6
        )


@pytest.mark.parametrize("serialize", [True, False])
def test_custom_cyclical_learning_rate_with_scale_mode(serialize):
    initial_learning_rate = 0.1
    maximal_learning_rate = 1
    step_size = 4000
    scale_mode = "iterations"

    def scale_fn(x):
        return 1 / (5 ** (x * 0.0001))

    custom_cyclical_lr = cyclical_learning_rate.CyclicalLearningRate(
        initial_learning_rate=initial_learning_rate,
        maximal_learning_rate=maximal_learning_rate,
        step_size=step_size,
        scale_fn=scale_fn,
        scale_mode=scale_mode,
    )
    custom_cyclical_lr = _maybe_serialized(custom_cyclical_lr, serialize)

    for step in range(1, 8001):
        cycle = np.floor(1 + step / (2 * step_size))
        non_bounded_value = np.abs(step / step_size - 2 * cycle + 1)
        expected = initial_learning_rate + (
            maximal_learning_rate - initial_learning_rate
        ) * np.maximum(0, 1 - non_bounded_value) * scale_fn(step)
        np.testing.assert_allclose(
            custom_cyclical_lr(step).numpy(), expected, 1e-6, 1e-6
        )
