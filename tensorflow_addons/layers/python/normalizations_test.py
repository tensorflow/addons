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
# =============================================================================

from tensorflow_addons.layers.python.normalizations import GroupNormalization,LayerNormalization,InstanceNormalization
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as keras
from tensorflow.python.training.rmsprop import RMSPropOptimizer

from tensorflow.python.platform import test
from tensorflow.python.framework import test_util as tf_test_util


class normalization_test(test.TestCase):

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weights(self):
        layer = GroupNormalization(groups=1,scale=False, center=False)
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

        layer = keras.layers.LayerNormalization()
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.weights), 2)




    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_groupnorm_flat(self):
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
        groups=[-1,16,1]
        for i in groups:
            model=keras.models.Sequential()
            model.add(GroupNormalization(
                 input_shape=(32,),groups=i))
            model.add(keras.layers.Dense(32))

            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            model.fit(
                    np.random.random((10,32)),
                    np.random.random((10,32)),
                    epochs=1,
                    batch_size=10)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))

    def test_groupnorm_conv(self):
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
        #groups=[1,5,-1]
        groups=[1]
        for i in groups:

            model = keras.models.Sequential()
            model.add(GroupNormalization(
                 input_shape=(20,20,3,),groups=i))

            model.add(keras.layers.Conv2D(5, (1, 1), padding='same'))

            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            model.fit(np.random.random((10,20, 20, 3)))
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))

    """def testUnknownShape(self):
        inputs = array_ops.placeholder(dtypes.float32)
        with self.assertRaisesRegexp(ValueError, 'undefined rank'):
            GroupNormalization(inputs)
            LayerNormaliztion(inputs)
            InstanceNormalization(inputs)"""
"""
class LayerNormalizationTest(keras_parameterized.TestCase):


  @tf_test_util.run_in_graph_and_eager_modes
  def test_layernorm_regularization(self):
    layer = keras.layers.LayerNormalization(
        gamma_regularizer='l1', beta_regularizer='l1')
    layer.build((None, 3, 4))
    self.assertEqual(len(layer.losses), 2)
    max_norm = keras.constraints.max_norm
    layer = keras.layers.LayerNormalization(
        gamma_constraint=max_norm, beta_constraint=max_norm)
    layer.build((None, 3, 4))
    self.assertEqual(layer.gamma.constraint, max_norm)
    self.assertEqual(layer.beta.constraint, max_norm)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_convnet(self):
    if test.is_gpu_available(cuda_only=True):
      with self.session(use_gpu=True):
        model = keras.models.Sequential()
        norm = keras.layers.LayerNormalization(input_shape=(3, 4, 4))
        model.add(norm)
        model.compile(loss='mse',
                      optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                      run_eagerly=testing_utils.should_run_eagerly())

        # centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
        model.fit(x, x, epochs=4, verbose=0)
        out = model.predict(x)
        out -= np.reshape(keras.backend.eval(norm.beta), (1, 3, 1, 1))
        out /= np.reshape(keras.backend.eval(norm.gamma), (1, 3, 1, 1))

        np.testing.assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
        np.testing.assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_convnet_channel_last(self):
    model = keras.models.Sequential()
    norm = keras.layers.LayerNormalization(input_shape=(4, 4, 3))
    model.add(norm)
    model.compile(loss='mse',
                  optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                  run_eagerly=testing_utils.should_run_eagerly())

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 4, 4, 3))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(keras.backend.eval(norm.beta), (1, 1, 1, 3))
    out /= np.reshape(keras.backend.eval(norm.gamma), (1, 1, 1, 3))

    np.testing.assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
    np.testing.assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_correctness(self):
    _run_layernorm_correctness_test(
        normalization.LayerNormalization, dtype='float32')

  @keras_parameterized.run_all_keras_modes
  def test_layernorm_mixed_precision(self):
    _run_layernorm_correctness_test(
        normalization.LayerNormalization, dtype='float16')

  def doOutputTest(self,
                   input_shape,
                   tol=1e-5,
                   norm_axis=None,
                   params_axis=-1,
                   dtype=None):
    ndim = len(input_shape)
    if norm_axis is None:
      moments_axis = range(1, ndim)
    elif isinstance(norm_axis, int):
      if norm_axis < 0:
        moments_axis = [norm_axis + ndim]
      else:
        moments_axis = [norm_axis]
    else:
      moments_axis = []
      for dim in norm_axis:
        if dim < 0:
          dim = dim + ndim
        moments_axis.append(dim)

    moments_axis = tuple(moments_axis)
    expected_shape = []
    for i in range(ndim):
      if i not in moments_axis:
        expected_shape.append(input_shape[i])

    expected_mean = np.zeros(expected_shape)
    expected_var = np.ones(expected_shape)
    for mu in [0.0, 1e2]:
      for sigma in [1.0, 0.1]:
        inputs = np.random.randn(*input_shape) * sigma + mu
        inputs_t = constant_op.constant(inputs, shape=input_shape)
        layer = normalization.LayerNormalization(
            norm_axis=norm_axis, params_axis=params_axis, dtype=dtype)
        outputs = layer(inputs_t)
        beta = layer.beta
        gamma = layer.gamma
        for weight in layer.weights:
          self.evaluate(weight.initializer)
        outputs = self.evaluate(outputs)
        beta = self.evaluate(beta)
        gamma = self.evaluate(gamma)

        # The mean and variance of the output should be close to 0 and 1
        # respectively.

        # Make sure that there are no NaNs
        self.assertFalse(np.isnan(outputs).any())
        mean = np.mean(outputs, axis=moments_axis)
        var = np.var(outputs, axis=moments_axis)
        # Layer-norm implemented in numpy
        eps = 1e-12
        expected_out = (
            (gamma * (inputs - np.mean(
                inputs, axis=moments_axis, keepdims=True)) /
             np.sqrt(eps + np.var(
                 inputs, axis=moments_axis, keepdims=True))) + beta)
        self.assertAllClose(expected_mean, mean, atol=tol, rtol=tol)
        self.assertAllClose(expected_var, var, atol=tol)
        # The full computation gets a bigger tolerance
        self.assertAllClose(expected_out, outputs, atol=5 * tol)

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput2DInput(self):
    self.doOutputTest((10, 300))
    self.doOutputTest((10, 300), norm_axis=[0])
    self.doOutputTest((10, 300), params_axis=[0, 1])

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput2DInputDegenerateNormAxis(self):
    with self.assertRaisesRegexp(ValueError, r'Invalid axis: 2'):
      self.doOutputTest((10, 300), norm_axis=2)

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput4DInput(self):
    self.doOutputTest((100, 10, 10, 3))

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutput4DInputNormOnInnermostAxis(self):
    # Equivalent tests
    shape = (100, 10, 10, 3)
    self.doOutputTest(
        shape, norm_axis=list(range(3, len(shape))), tol=1e-4, dtype='float64')
    self.doOutputTest(shape, norm_axis=-1, tol=1e-4, dtype='float64')

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputSmallInput(self):
    self.doOutputTest((10, 10, 10, 30))

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputSmallInputNormOnInnermostAxis(self):
    self.doOutputTest((10, 10, 10, 30), norm_axis=3)

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputSmallInputNormOnMixedAxes(self):
    self.doOutputTest((10, 10, 10, 30), norm_axis=[0, 3])
    self.doOutputTest((10, 10, 10, 30), params_axis=[-2, -1])
    self.doOutputTest((10, 10, 10, 30), norm_axis=[0, 3],
                      params_axis=[-3, -2, -1])

  @tf_test_util.run_in_graph_and_eager_modes
  def testOutputBigInput(self):
    self.doOutputTest((1, 100, 100, 1))
    self.doOutputTest((1, 100, 100, 1), norm_axis=[1, 2])
    self.doOutputTest((1, 100, 100, 1), norm_axis=[1, 2],
                      params_axis=[-2, -1])

"""
if __name__ == "__main__":
    test.main()
