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
"""Tests NoisyDense layer."""


import pytest
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow_addons.utils import test_utils
from tensorflow_addons.layers.noisy_dense import NoisyDense


@pytest.mark.parametrize(
    "input_shape", [(3, 2), (3, 4, 2), (None, None, 2), (3, 4, 5, 2)]
)
def test_noisy_dense(input_shape):
    test_utils.layer_test(NoisyDense, kwargs={"units": 3}, input_shape=input_shape)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_noisy_dense_dtype(dtype):
    inputs = tf.convert_to_tensor(
        np.random.randint(low=0, high=7, size=(2, 2)), dtype=dtype
    )
    layer = NoisyDense(5, dtype=dtype, name="noisy_dense_" + dtype)
    outputs = layer(inputs)
    np.testing.assert_array_equal(outputs.dtype, dtype)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_with_policy():
    inputs = tf.convert_to_tensor(np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(5, dtype="mixed_float16", name="noisy_dense_policy")
    outputs = layer(inputs)
    output_signature = layer.compute_output_signature(
        tf.TensorSpec(dtype="float16", shape=(2, 2))
    )
    np.testing.assert_array_equal(output_signature.dtype, tf.dtypes.float16)
    np.testing.assert_array_equal(output_signature.shape, (2, 5))
    np.testing.assert_array_equal(outputs.dtype, "float16")
    np.testing.assert_array_equal(layer.mu_kernel.dtype, "float32")
    np.testing.assert_array_equal(layer.sigma_kernel.dtype, "float32")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_regularization():
    layer = NoisyDense(
        3,
        kernel_regularizer=keras.regularizers.l1(0.01),
        bias_regularizer="l1",
        activity_regularizer="l2",
        name="noisy_dense_reg",
    )
    layer(keras.backend.variable(np.ones((2, 4))))
    np.testing.assert_array_equal(5, len(layer.losses))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_constraints():
    k_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    layer = NoisyDense(
        3,
        kernel_constraint=k_constraint,
        bias_constraint=b_constraint,
        name="noisy_dense_constriants",
    )
    layer(keras.backend.variable(np.ones((2, 4))))
    np.testing.assert_array_equal(layer.mu_kernel.constraint, k_constraint)
    np.testing.assert_array_equal(layer.sigma_kernel.constraint, k_constraint)
    np.testing.assert_array_equal(layer.mu_bias.constraint, b_constraint)
    np.testing.assert_array_equal(layer.sigma_bias.constraint, b_constraint)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_reset_noise():
    inputs = tf.convert_to_tensor(np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(5, name="noise_dense_auto_reset_noise")
    layer(inputs)
    initial_eps_kernel = tf.identity(layer.eps_kernel)
    initial_eps_bias = tf.identity(layer.eps_bias)
    layer.reset_noise()
    layer(inputs)
    new_eps_kernel = layer.eps_kernel
    new_eps_bias = layer.eps_bias
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        initial_eps_kernel,
        new_eps_kernel,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        initial_eps_bias,
        new_eps_bias,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_reset_noise_independent():
    inputs = tf.convert_to_tensor(np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(
        5, use_factorised=False, name="noise_dense_auto_reset_noise_independent"
    )
    layer(inputs)
    initial_eps_kernel = tf.identity(layer.eps_kernel)
    initial_eps_bias = tf.identity(layer.eps_bias)
    layer.reset_noise()
    layer(inputs)
    new_eps_kernel = layer.eps_kernel
    new_eps_bias = layer.eps_bias
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        initial_eps_kernel,
        new_eps_kernel,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        initial_eps_bias,
        new_eps_bias,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_remove_noise():
    inputs = tf.convert_to_tensor(np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(5, name="noise_dense_manual_reset_noise")
    layer(inputs)
    initial_eps_kernel = tf.identity(layer.eps_kernel)
    initial_eps_bias = tf.identity(layer.eps_bias)
    layer.remove_noise()
    layer(inputs)
    new_eps_kernel = layer.eps_kernel
    new_eps_bias = layer.eps_bias
    kernel_zeros = tf.zeros(initial_eps_kernel.shape, dtype=initial_eps_kernel.dtype)
    bias_zeros = tf.zeros(initial_eps_bias.shape, dtype=initial_eps_kernel.dtype)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        initial_eps_kernel,
        new_eps_kernel,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        initial_eps_bias,
        new_eps_bias,
    )
    np.testing.assert_array_equal(kernel_zeros, new_eps_kernel)
    np.testing.assert_array_equal(bias_zeros, new_eps_bias)
