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
"""Tests NoisyDense layer."""


import numpy as np
import pytest
from tensorflow.python import keras
from tensorflow_addons.utils import test_utils
from tensorflow_addons.layers.noisy_dense import NoisyDense
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.framework import tensor_spec
from tensorflow.debugging import assert_equal
from tensorflow.python.keras.mixed_precision.experimental import policy

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_noisy_dense(dtype):
  test_utils.layer_test(
      NoisyDense, kwargs={'units': 3, "dtype": dtype}, input_shape=(3, 2))

  test_utils.layer_test(
      NoisyDense, kwargs={'units': 3, "dtype": dtype}, input_shape=(3, 4, 2))

  test_utils.layer_test(
      NoisyDense, kwargs={'units': 3, "dtype": dtype}, input_shape=(None, None, 2))

  test_utils.layer_test(
      NoisyDense, kwargs={'units': 3, "dtype": dtype}, input_shape=(3, 4, 5, 2))

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_with_policy():
  inputs = ops.convert_to_tensor_v2(
      np.random.randint(low=0, high=7, size=(2, 2)))
  layer = NoisyDense(5, dtype=policy.Policy('mixed_float16'))
  outputs = layer(inputs)
  output_signature = layer.compute_output_signature(
      tensor_spec.TensorSpec(dtype='float16', shape=(2, 2)))
  assert_equal(output_signature.dtype, dtypes.float16)
  assert_equal(output_signature.shape, (2, 5))
  sassert_equal(outputs.dtype, 'float16')
  assert_equal(layer.kernel.dtype, 'float32')

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_regularization(self):
  layer = NoisyDense(
      3,
      kernel_regularizer=keras.regularizers.l1(0.01),
      bias_regularizer='l1',
      activity_regularizer='l2',
      name='noisy_dense_reg')
  layer(keras.backend.variable(np.ones((2, 4))))
  assert_equal(3, len(layer.losses))

@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_noisy_dense_constraints(self):
  k_constraint = keras.constraints.max_norm(0.01)
  b_constraint = keras.constraints.max_norm(0.01)
  layer = NoisyDense(
      3, kernel_constraint=k_constraint, bias_constraint=b_constraint)
  layer(keras.backend.variable(np.ones((2, 4))))
  assert_equal(layer.kernel.constraint, k_constraint)
  assert_equal(layer.bias.constraint, b_constraint)
