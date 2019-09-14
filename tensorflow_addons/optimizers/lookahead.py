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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import ops
from tensorflow.python.ops import math_ops, state_ops, control_flow_ops
from tensorflow.python.keras import optimizers
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class Lookahead(tf.keras.optimizers.Optimizer):
    """This class allows to extend optimizers with the lookahead mechanism.

    The mechanism is proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1). The optimizer iteratively updates two
    sets of weights: the search directions for weights are chosen by the inner
    optimizer, while the "slow weights" are updated each `k` steps based on the
    directions of the "fast weights" and the two sets of weights are
    synchronized. This method improves the learning stability and lowers the
    variance of its inner optimizer.

    Example of usage:

    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.Lookahead(opt)
    ```
    """

    def __init__(self,
                 optimizer,
                 k=6,
                 alpha=0.5,
                 name="Lookahead",
                 **kwargs):
        r"""Wrap optimizer with the lookahead mechanism.

        Args:
            optimizer: A Tensor or a floating point value.
                The learning rate.
            k: An integer. Synchronization period of lookahead.
                Enable lookahead mechanism by setting it with a positive value.
            alpha: A float value. Slow weights step size.
            name: Optional name for the operations created when applying
                gradients. Defaults to "RectifiedAdam".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
                `lr`, `decay`}. `clipnorm` is clip gradients by norm;
                `clipvalue` is clip gradients by value, `decay` is included for
                backward compatibility to allow time inverse decay of learning
                rate. `lr` is included for backward compatibility, recommended
                to use `learning_rate` instead.
        """
        super(Lookahead, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer")

        self._optimizer = optimizer
        self._set_hyper('k', k)
        self._set_hyper('alpha', alpha)
        self._initialized = False

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'slow')

    def apply_gradients(self, grads_and_vars, name=None):
        var_list = [v for (_, v) in grads_and_vars]

        with tf.keras.backend.name_scope(self._name):
            with ops.init_scope():
                _ = self.iterations
                self._create_hypers()
                self._create_slots(var_list)
            self._prepare(var_list)

        if self._initialized:
            init_op = tf.no_op()
        else:
            self._initialized = True
            init_op = self._init_op(var_list)

        with tf.control_dependencies([init_op]):
            train_op = self._optimizer.apply_gradients(
                grads_and_vars, name=name)
            with tf.control_dependencies([train_op]):
                lookahead_op = control_flow_ops.group([
                    self._look_ahead_op(var) for var in var_list])

        return control_flow_ops.group(init_op, train_op, lookahead_op)

    def _init_op(self, var_list):
        updates = []
        for var in var_list:
            slow_var = self.get_slot(var, 'slow')
            updates.append(state_ops.assign(
                slow_var,
                tf.where(
                    math_ops.equal(self.iterations,
                                   tf.constant(0, dtype=self.iterations.dtype)),
                    var,
                    slow_var,
                ),
                use_locking=self._use_locking))
        return control_flow_ops.group(*updates)

    def _look_ahead_op(self, var):
        var_dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, 'slow')
        local_step = math_ops.cast(self.iterations, var_dtype)
        k = self._get_hyper('k', local_step.dtype)
        alpha = self._get_hyper('alpha', var_dtype)
        step_back = slow_var + alpha * (var - slow_var)
        sync_cond = math_ops.equal(local_step % k, 0)
        with tf.control_dependencies([step_back]):
            slow_update = state_ops.assign(slow_var, tf.where(
                sync_cond,
                step_back,
                slow_var,
            ), use_locking=self._use_locking)
            var_update = state_ops.assign(var, tf.where(
                sync_cond,
                step_back,
                var,
            ), use_locking=self._use_locking)
        return control_flow_ops.group(slow_update, var_update)

    @property
    def iterations(self):
        return self._optimizer.iterations

    @property
    def weights(self):
        return self._optimizer.weights

    def _resource_apply_dense(self, grad, var):
        return self._optimizer._resource_apply_dense(grad, var)  # pylint: disable=protected-access

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        return self._optimizer._resource_apply_sparse_duplicate_indices(  # pylint: disable=protected-access
            grad, var, indices)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._optimizer._resource_apply_sparse(grad, var, indices)  # pylint: disable=protected-access

    def get_config(self):
        config = {
            'optimizer': optimizers.serialize(self._optimizer),
            'k': self._serialize_hyperparameter('k'),
            'alpha': self._serialize_hyperparameter('alpha'),
        }
        base_config = super(Lookahead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)
