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
"""Conditional Gradient method for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class ConditionalGradient(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Conditional Gradient optimization.

    This optimizer helps handle constraints well.

    Currently only supports frobenius norm constraint.
    See https://arxiv.org/pdf/1803.06453.pdf

    ```
    variable -= (1-learning_rate) * (variable + lambda_ * gradient
        / (frobenius_norm(gradient) + epsilon))
    ```

    Note that we choose "lambda_" here to refer to the constraint "lambda" in
    the paper.
    And 'epsilon' is constant with tiny value as compared to the value of
    frobenius_norm of gradient. The purpose of 'epsilon' here is to avoid the
    case that the value of frobenius_norm of gradient is 0.

    In this implementation, we choose 'epsilon' with value of 10^-7.
    """

    def __init__(self,
                 learning_rate,
                 lambda_,
                 epsilon=1e-7,
                 use_locking=False,
                 name='ConditionalGradient',
                 **kwargs):
        """Construct a conditional gradient optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value.
                        The learning rate.
            lambda_: A `Tensor` or a floating point value. The constraint.
            epsilon: A `Tensor` or a floating point value. A small constant
                    for numerical stability when handling the case of norm of
                    gradient to be zero.
            use_locking: If `True` use locks for update operations.
            name: Optional name prefix for the operations created when
                applying gradients.  Defaults to 'ConditionalGradient'
        """
        super(ConditionalGradient, self).__init__(name=name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('lambda_', lambda_)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self._set_hyper('use_locking', use_locking)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'lambda_': self._serialize_hyperparameter('lambda_'),
            'epsilon': self.epsilon,
            'use_locking': self._serialize_hyperparameter('use_locking')
        }
        base_config = super(ConditionalGradient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, 'conditional_gradient')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(ConditionalGradient, self)._prepare_local(
            var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]['learning_rate'] = tf.identity(
            self._get_hyper('learning_rate', var_dtype))
        apply_state[(var_device, var_dtype)]['lambda_'] = tf.identity(
            self._get_hyper('lambda_', var_dtype))
        apply_state[(var_device, var_dtype)]['epsilon'] = tf.convert_to_tensor(
            self.epsilon, var_dtype)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        def frobenius_norm(m):
            return tf.math.reduce_sum(m**2)**0.5

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        norm = tf.convert_to_tensor(
            frobenius_norm(grad), name='norm', dtype=var.dtype.base_dtype)
        lr = coefficients['learning_rate']
        lambda_ = coefficients['lambda_']
        epsilon = coefficients['epsilon']
        var_update_tensor = (tf.math.multiply(var, lr) -
                             (1 - lr) * lambda_ * grad / (norm + epsilon))
        var_update_kwargs = {
            'resource': var.handle,
            'value': var_update_tensor,
        }
        var_update_op = tf.raw_ops.AssignVariableOp(**var_update_kwargs)
        return tf.group(var_update_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        def frobenius_norm(m):
            return tf.reduce_sum(m**2)**0.5

        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        norm = tf.convert_to_tensor(
            frobenius_norm(grad), name='norm', dtype=var.dtype.base_dtype)
        lr = coefficients['learning_rate']
        lambda_ = coefficients['lambda_']
        epsilon = coefficients['epsilon']
        var_slice = tf.gather(var, indices)
        var_update_value = (tf.math.multiply(var_slice, lr) -
                            (1 - lr) * lambda_ * grad / (norm + epsilon))
        var_update_kwargs = {
            'resource': var.handle,
            'indices': indices,
            'updates': var_update_value
        }
        var_update_op = tf.raw_ops.ResourceScatterUpdate(**var_update_kwargs)
        return tf.group(var_update_op)
