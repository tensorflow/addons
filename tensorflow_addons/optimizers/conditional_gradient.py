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
    Helps handle constraints well.
    Currently only supports frobenius norm constraint.
    See https://arxiv.org/pdf/1803.06453.pdf
    ```
    variable -= (1-learning_rate)
        * (variable + Lambda * gradient / frobenius_norm(gradient))
    ```
    """

    def __init__(self,
                 learning_rate,
                 Lambda,
                 use_locking=False,
                 name="ConditionalGradient",
                 **kwargs):
        """Construct a conditional gradient optimizer.
            Args:
            learning_rate: A `Tensor` or a floating point value.
                           The learning rate.
            Lambda: A `Tensor` or a floating point value. The constraint.
            use_locking: If `True` use locks for update operations.
            name: Optional name prefix for the operations created when
                  applying gradients.  Defaults to "ConditionalGradient"
        """
        super(ConditionalGradient, self).__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("Lambda", Lambda)
        self._set_hyper("use_locking", use_locking)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'Lambda': self._serialize_hyperparameter('Lambda'),
            'use_locking': self._serialize_hyperparameter('use_locking')
        }
        base_config = super(ConditionalGradient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "conditional_gradient")

    def _prepare(self, var_list):
        learning_rate = self._get_hyper('learning_rate')
        if callable(learning_rate):
            learning_rate = learning_rate()
        self._learning_rate_tensor = tf.convert_to_tensor(
            learning_rate, name="learning_rate")
        Lambda = self._get_hyper('Lambda')
        if callable(Lambda):
            Lambda = Lambda()
        self._Lambda_tensor = tf.convert_to_tensor(Lambda, name="Lambda")
        return super(ConditionalGradient, self)._prepare(var_list)

    def _resource_apply_dense(self, grad, var):
        def frobenius_norm(m):
            return tf.math.reduce_sum(m**2)**0.5

        norm = tf.convert_to_tensor(
            frobenius_norm(grad), name="norm", dtype=var.dtype.base_dtype)
        lr = tf.dtypes.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        Lambda = tf.dtypes.cast(self._Lambda_tensor, var.dtype.base_dtype)
        var_update_tensor = (
            tf.math.multiply(var, lr) - (1 - lr) * Lambda * grad / norm)
        var_update_kwargs = {
            'resource': var.handle,
            'value': var_update_tensor,
        }

        var_update_op = tf.raw_ops.AssignVariableOp(**var_update_kwargs)
        return tf.group(var_update_op)

    def _resource_apply_sparse(self, grad, var, indices):
        def frobenius_norm(m):
            return tf.reduce_sum(m**2)**0.5

        norm = tf.convert_to_tensor(
            frobenius_norm(grad), name="norm", dtype=var.dtype.base_dtype)
        lr = tf.dtypes.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        Lambda = tf.dtypes.cast(self._Lambda_tensor, var.dtype.base_dtype)
        var_slice = tf.gather(var, indices)
        var_update_value = (
            tf.math.multiply(var_slice, lr) - (1 - lr) * Lambda * grad / norm)
        var_update_kwargs = {
            'resource': var.handle,
            'indices': indices,
            'updates': var_update_value
        }
        var_update_op = tf.raw_ops.ResourceScatterUpdate(**var_update_kwargs)
        return tf.group(var_update_op)
