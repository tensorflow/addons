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
"""Novograd for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training import training_ops


@tf.keras.utils.register_keras_serializable(package='Addons')
class Novograd(tf.keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=1,
                 beta_1=0.95,
                 beta_2=0.98,
                 epsilon=1e-8,
                 weight_decay=0.0,
                 grad_averaging=False,
                 name='Novograd',
                 **kwargs):
        super(Novograd, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('grad_averaging', grad_averaging)
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var=var, slot_name='m', initializer='zeros')
        for var in var_list:
            self.add_slot(
                var=var,
                slot_name='v',
                initializer=tf.zeros(shape=[], dtype=var.dtype))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(Novograd, self)._prepare_local(var_device, var_dtype,
                                             apply_state)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
              (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                one_minus_beta_2_t=1 - beta_2_t,
            ))

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(Novograd, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        weight_decay = self._get_hyper('weight_decay')
        grad_averaging = self._get_hyper('grad_averaging')

        v = self.get_slot(var, 'v')
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        v_t = tf.cond(tf.equal(self.iterations, 0),
                      lambda: g_2,
                      lambda: v * coefficients['beta_2_t'] + g_2 * coefficients['one_minus_beta_2_t'])
        v_t = v.assign(v_t, use_locking=self._use_locking)

        grad = grad / (tf.sqrt(v_t) + self.epsilon)
        grad = tf.cond(grad_averaging,
                       lambda: grad * coefficients['one_minus_beta_1_t'],
                       lambda: grad)
        grad = tf.cond(tf.greater(weight_decay, 0),
                       lambda: grad + weight_decay * var,
                       lambda: grad)
        m = self.get_slot(var, 'm')
        return training_ops.resource_apply_momentum(
            var.handle,
            m.handle,
            coefficients['lr'],
            grad,
            coefficients['beta_1_t'],
            use_locking=self._use_locking,
            use_nesterov=False)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        weight_decay = self._get_hyper('weight_decay')
        grad_averaging = self._get_hyper('grad_averaging')

        v = self.get_slot(var, 'v')
        g_2 = tf.sparse.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        # v is just a scalar and does not need to involve sparse tensors.
        v_t = tf.cond(tf.equal(self.iterations, 0),
                      lambda: g_2,
                      lambda: v * coefficients['beta_2_t'] + g_2 * coefficients['one_minus_beta_2_t'])
        v_t = v.assign(v_t, use_locking=self._use_locking)

        grad = grad / (tf.sqrt(v_t) + self.epsilon)
        grad = tf.cond(grad_averaging,
                       lambda: grad * coefficients['one_minus_beta_1_t'],
                       lambda: grad)
        grad = tf.cond(tf.greater(weight_decay, 0),
                       self._resource_scatter_add(grad, indices, weight_decay * var),
                       grad)
        m = self.get_slot(var, 'm')
        return training_ops.resource_apply_sparse_momentum(
            var.handle,
            m.handle,
            coefficients['lr'],
            grad,
            indices,
            coefficients['beta_1_t'],
            use_locking=self._use_locking,
            use_nesterov=False)

    def get_config(self):
        config = super(Novograd, self).get_config()
        config.update({
            'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
            'beta_1':
            self._serialize_hyperparameter('beta_1'),
            'beta_2':
            self._serialize_hyperparameter('beta_2'),
            'epsilon':
            self.epsilon,
            'weight_decay':
            self._serialize_hyperparameter('weight_decay'),
            'grad_averaging':
            self._serialize_hyperparameter('grad_averaging'),
        })
        return config
