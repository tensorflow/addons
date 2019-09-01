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
    #class ConditionalGradient(tf.keras.optimizer_v2.OptimizerV2):
class ConditionalGradient(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Conditional Gradient optimization.
    Helps handle constraints well.
    Currently only supports frobenius norm constraint.
    See https://arxiv.org/pdf/1803.06453.pdf
    ```
    variable -= (1-learning_rate)
        * (variable + lamda * gradient / frobenius_norm(gradient))
    ```
    """

    def __init__(self, learning_rate, lamda,
                use_locking=False, name="ConditionalGradient"):
        """Construct a conditional gradient optimizer.
            Args:
            learning_rate: A `Tensor` or a floating point value.
                           The learning rate.
            lamda: A `Tensor` or a floating point value. The constraint.
            use_locking: If `True` use locks for update operations.
            name: Optional name prefix for the operations created when
                  applying gradients.  Defaults to "ConditionalGradient"
        """
        super(ConditionalGradient, self).__init__(name=name)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("lamda", lamda)

    def get_config(self):
        config = {
            'learning_rate': self._learning_rate,
            'lamda': self._lamda,
            'use_locking': self._use_locking
        }
        base_config = super(ConditionalGradient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "conditional_gradient")

    def _prepare(self, var_list):
        learning_rate = self.learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate()
        self._learning_rate_tensor = tf.convert_to_tensor(
                    learning_rate, name="learning_rate")
        lamda = self.lamda
        if callable(lamda):
            lamda = lamda()
        self._lamda_tensor = tf.convert_to_tensor(lamda, name="lamda")

    def _resource_apply_dense(self, grad, var):
        def frobenius_norm(m):
            return tf.math.reduce_sum(m ** 2) ** 0.5
        norm = tf.convert_to_tensor(frobenius_norm(grad), name="norm")
        norm = tf.dtypes.cast(norm, var.dtype.base_dtype)
        lr = tf.dtypes.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        lamda = tf.dtypes.cast(self._lamda_tensor, var.dtype.base_dtype)
        var_update_tensor = (tf.math.multiply(var, lr) -
                    (1 - lr) * lamda * grad / norm)
        var_update_kwargs = {
            'resource': var.handle,
            'value': var_update_tensor,
        }

        var_update_op = tf.raw_ops.AssignVariableOp(**var_update_kwargs)
        return tf.group(var_update_op)

    def _resource_apply_sparse(self, grad, var, indices):
        def frobenius_norm(m):
            return tf.reduce_sum(m ** 2) ** 0.5
        norm = tf.convert_to_tensor(frobenius_norm(grad), name="norm")
        norm = tf.dtypes.cast(norm, var.dtype.base_dtype)
        lr = tf.dtypes.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        lamda = tf.dtypes.cast(self._lamda_tensor, var.dtype.base_dtype)
        var_slice = tf.gather(var, indices)
        var_update_value = (tf.math.multiply(var_slice, lr) -
                    (1 - lr) * lamda * grad / norm)
        var_update_kwargs = {
            'resource': var.handle,
            'indices': indices,
            'updates': var_update_value
        }
        var_update_op = tf.raw_ops.ResourceScatterUpdate(**var_update_kwargs)
        return tf.group(var_update_op)
