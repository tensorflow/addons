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

import tensorflow as tf

from tensorflow.keras import regularizers, constraints, initializers
from tensorflow.keras.layers import Dense, InputSpec 
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable(package="Addons")
class NoisyDense(Dense):
  """Densely-connected NN layer with additive zero-centered Gaussian noise (NoisyNet).
  `NoisyDense` implements the operation:
  `output = activation(dot(input, kernel + kernel_sigma * kernel_epsilon) + bias + bias_sigma * bias_epsilon)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a base weights matrix
  created by the layer, `kernel_sigma` is a noise weights matrix
  created by the layer, `bias` is a base bias vector created by the layer, 
  `bias_sigma` is a noise bias vector created by the layer,
  'kernel_epsilon' and 'bias_epsilon' are noise random variables.
  (biases are only applicable if `use_bias` is `True`)
  
  There are implemented both variants: 
    1. Independent Gaussian noise                                    
    2. Factorised Gaussian noise.
  We can choose between that by 'use_factorised' parameter.
  Arguments:
    units: Positive integer, dimensionality of the output space.
    sigma0: Float, initial sigma parameter (uses only if use_factorised=True)
    use_factorised: Boolean, whether the layer uses independent or factorised Gaussian noise
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    kernel_sigma_regularizer: Regularizer function applied to
      the `kernel_sigma` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    bias_sigma_regularizer: Regularizer function applied to the bias_sigma vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    kernel_sigma_constraint: Constraint function applied to
      the `kernel_sigma` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    bias_sigma_constraint: Constraint function applied to the bias_sigma vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  Reference:
    - [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
  """

  def __init__(self,
               units,
               sigma0=0.5,
               use_factorised=True,
               activation=None, 
               use_bias=True, 
               kernel_initializer='glorot_uniform', 
               bias_initializer='zeros',
               kernel_regularizer=None, 
               kernel_sigma_regularizer=None,
               bias_regularizer=None, 
               bias_sigma_regularizer=None,
               activity_regularizer=None, 
               kernel_constraint=None, 
               kernel_sigma_constraint=None,
               bias_constraint=None, 
               bias_sigma_constraint=None,
               **kwargs):
    super(NoisyDense, self).__init__(units=units, 
                                     activation=activation, 
                                     use_bias=use_bias, 
                                     kernel_initializer=kernel_initializer, 
                                     bias_initializer=bias_initializer, 
                                     kernel_regularizer=kernel_regularizer, 
                                     bias_regularizer=bias_regularizer, 
                                     activity_regularizer=activity_regularizer, 
                                     kernel_constraint=kernel_constraint, 
                                     bias_constraint=bias_constraint, 
                                     **kwargs)

    self.sigma0 = sigma0
    self.use_factorised = use_factorised
    
    self.kernel_sigma_regularizer = regularizers.get(kernel_sigma_regularizer)
    self.bias_sigma_regularizer = regularizers.get(bias_sigma_regularizer)
    self.kernel_sigma_constraint = constraints.get(kernel_sigma_constraint)
    self.bias_sigma_constraint = constraints.get(bias_sigma_constraint)

  def build(self, input_shape):
    dtype = tf.dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tf.TensorShape(input_shape)
    last_dim = tf.compat.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel_mu = self.add_weight(
        'kernel_mu',
        shape=[last_dim, self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias_mu = self.add_weight(
          'bias_mu',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

    # use factorising Gaussian variables
    if self.use_factorised:
      sigma_init = self.sigma0 / tf.sqrt(self.kernel_mu.shape[0])
    # use independent Gaussian variables  
    else:
      sigma_init = 0.017
    
    # create sigma weights
    self.kernel_sigma = self.add_weight(
        'kernel_sigma',
        shape=self.kernel_mu.shape,
        initializer=initializers.Constant(value=sigma_init),
        regularizer=self.kernel_sigma_regularizer,
        constraint=self.kernel_sigma_constraint,
        dtype=self.dtype,
        trainable=True)
    if self.use_bias:
      self.bias_sigma = self.add_weight(
          'bias_sigma',
          shape=self.bias.shape,
          initializer=initializers.Constant(value=sigma_init),
          regularizer=self.bias_sigma_regularizer,
          constraint=self.bias_sigma_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias_sigma = None
    
    # create noise variables
    self.kernel_epsilon = self.add_weight(
          name='kernel_epsilon',
          shape=self.kernel_mu.shape,
          dtype=self.dtype,
          initializer='zeros',
          trainable=False)
    if self.use_bias:
      self.bias_epsilon = self.add_weight(
            name='bias_epsilon',
            shape=self.bias.shape,
            dtype=self.dtype,
            initializer='zeros',
            trainable=False)
    else:
      self.bias_epsilon = None

    # init epsilon parameters
    self.reset_noise()
    self.built = True

  def call(self, inputs):
    self.kernel = tf.add(self.kernel_mu, tf.mul(self.kernel_sigma, self.kernel_epsilon))
    self.bias = self.bias_mu
    if self.bias is not None:
      self.bias = tf.add(self.bias_mu, tf.mul(self.bias_sigma, self.bias_epsilon))
    
    return super().call(inputs)

  def get_config(self):
    config = super(NoisyDense, self).get_config()
    config.update({
        'sigma0':
            self.sigma0,
        'use_factorised':
            self.use_factorised,
        'kernel_sigma_regularizer':
            regularizers.serialize(self.kernel_sigma_regularizer),
        'bias_sigma_regularizer':
            regularizers.serialize(self.bias_sigma_regularizer),
        'kernel_sigma_constraint':
            constraints.serialize(self.kernel_sigma_constraint),
        'bias_sigma_constraint':
            constraints.serialize(self.bias_sigma_constraint)
    })
    return config
  
  def _scale_noise(self, size):
    x = K.random_normal(shape=size,
                        mean=0.0,
                        stddev=1.0,
                        dtype=self.dtype)
    return tf.mul(tf.sign(x), tf.sqrt(tf.abs(x)))
    
  def reset_noise(self):
    if self.use_factorised:
      in_eps = self._scale_noise((self.kernel_epsilon.shape[0], 1))
      out_eps = self._scale_noise((1, self.units))
      w_eps = tf.matmul(in_eps, out_eps)
      b_eps = out_eps[0]
    else:
      # generate independent variables
      w_eps = K.random_normal(shape=self.kernel_epsilon.shape,
                              mean=0.0,
                              stddev=1.0,
                              dtype=self.dtype)
      b_eps = K.random_normal(shape=self.bias_epsilon.shape,
                              mean=0.0,
                              stddev=1.0,
                              dtype=self.dtype)

    self.kernel_epsilon.assign(w_eps)
    self.bias_epsilon.assign(b_eps)
    
  def remove_noise(self):
    self.kernel_epsilon.assign(tf.zeros(self.kernel_epsilon.shape, self.dtype))
    self.bias_epsilon.assign(tf.zeros(self.bias_epsilon.shape, self.dtype))
