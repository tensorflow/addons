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
from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow_addons.layers.noisy_dense import NoisyDense
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.framework import tensor_spec


@keras_parameterized.run_all_keras_modes
class NoisyDenseTest(keras_parameterized.TestCase):
  def test_noisy_dense(self):
    testing_utils.layer_test(
        NoisyDense, kwargs={'units': 3}, input_shape=(3, 2))

    testing_utils.layer_test(
        NoisyDense, kwargs={'units': 3}, input_shape=(3, 4, 2))

    testing_utils.layer_test(
        NoisyDense, kwargs={'units': 3}, input_shape=(None, None, 2))

    testing_utils.layer_test(
        NoisyDense, kwargs={'units': 3}, input_shape=(3, 4, 5, 2))

  def test_noisy_dense_dtype(self):
    inputs = ops.convert_to_tensor_v2(
        np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(5, dtype='float32')
    outputs = layer(inputs)
    self.assertEqual(outputs.dtype, 'float32')

  def test_noisy_dense_with_policy(self):
    inputs = ops.convert_to_tensor_v2(
        np.random.randint(low=0, high=7, size=(2, 2)))
    layer = NoisyDense(5, dtype=policy.Policy('mixed_float16'))
    outputs = layer(inputs)
    output_signature = layer.compute_output_signature(
        tensor_spec.TensorSpec(dtype='float16', shape=(2, 2)))
    self.assertEqual(output_signature.dtype, dtypes.float16)
    self.assertEqual(output_signature.shape, (2, 5))
    self.assertEqual(outputs.dtype, 'float16')
    self.assertEqual(layer.kernel.dtype, 'float32')

  def test_noisy_dense_regularization(self):
    layer = NoisyDense(
        3,
        kernel_regularizer=keras.regularizers.l1(0.01),
        bias_regularizer='l1',
        activity_regularizer='l2',
        name='noisy_dense_reg')
    layer(keras.backend.variable(np.ones((2, 4))))
    self.assertEqual(3, len(layer.losses))

  def test_noisy_dense_constraints(self):
    k_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    layer = NoisyDense(
        3, kernel_constraint=k_constraint, bias_constraint=b_constraint)
    layer(keras.backend.variable(np.ones((2, 4))))
    self.assertEqual(layer.kernel.constraint, k_constraint)
    self.assertEqual(layer.bias.constraint, b_constraint)
