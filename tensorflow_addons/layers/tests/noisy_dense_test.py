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
from tensorflow.keras.mixed_precision.experimental import Policy

from tensorflow_addons.utils import test_utils
from tensorflow_addons.layers.noisy_dense import NoisyDense


def test_noisy_dense():
    test_utils.layer_test(
        NoisyDense, kwargs={'units': 3, 'sigma0': 0.4, 'use_factorised': True}, input_shape=(3, 2))

    test_utils.layer_test(
        NoisyDense, kwargs={'units': 3, 'sigma0': 0.4, 'use_factorised': False}, input_shape=(3, 4, 2))

    test_utils.layer_test(
        NoisyDense, kwargs={'units': 3, 'sigma0': 0.4, 'use_factorised': True}, input_shape=(None, None, 2))

    test_utils.layer_test(
        NoisyDense, kwargs={'units': 3, 'sigma0': 0.4, 'use_factorised': False}, input_shape=(3, 4, 5, 2))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_noisy_dense_dtype(dtype):
    inputs = tf.convert_to_tensor(
        np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(5, sigma0=0.4, dtype=dtype)
    outputs = layer(inputs)    
    layer.remove_noise()
    np.testing.assert_array_equal(outputs.dtype, dtype)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_regularization():
    layer = NoisyDense(
        3,
        kernel_regularizer=keras.regularizers.l1(0.01),
        bias_regularizer='l1',
        activity_regularizer='l2',
        kernel_sigma_regularizer='l1',
        bias_sigma_regularizer='l2',
        name='dense_reg')
    layer(keras.backend.variable(np.ones((2, 4))))
    np.testing.assert_array_equal(5, len(layer.losses))

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_constraints():
    k_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    layer = NoisyDense(
        3, 
        sigma0=0.8, 
        use_factorised=False, 
        kernel_constraint=k_constraint, 
        kernel_sigma_constraint=k_constraint, 
        bias_constraint=b_constraint, 
        bias_sigma_constraint=b_constraint)
    layer(keras.backend.variable(np.ones((2, 4))))
    np.testing.assert_array_equal(layer.kernel_mu.constraint, k_constraint)
    np.testing.assert_array_equal(layer.bias_mu.constraint, b_constraint)
    np.testing.assert_array_equal(layer.kernel_sigma.constraint, k_constraint)
    np.testing.assert_array_equal(layer.bias_sigma.constraint, b_constraint)
