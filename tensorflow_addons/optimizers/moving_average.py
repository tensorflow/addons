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
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class MovingAverage(tf.keras.optimizers.Optimizer):
    """Optimizer that computes a moving average of the variables.

    Empirically it has been found that using the moving average of the trained
    parameters of a deep network is better than using its trained parameters
    directly. This optimizer allows you to compute this moving average and swap
    the variables at save time so that any code outside of the training loop
    will use by default the average values instead of the original ones.

    Example of usage:

    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.MovingAverage(opt)

    ```
    """

    def __init__(self,
                 optimizer,
                 average_decay=0.1,
                 num_updates=None,
                 sequential_update=True,
                 name="MovingAverage",
                 **kwargs):

        super(MovingAverage, self).__init__(name, **kwargs)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer")

        if num_updates is not None and not isinstance(num_updates, int):
            raise TypeError("num_updates must be None or of integer type")

        if not isinstance(sequential_update, bool):
            raise TypeError("sequential_update must be of bool type")

        self._optimizer = optimizer

        with tf.name_scope(name):
            self._ema = tf.train.ExponentialMovingAverage(
                average_decay, num_updates=num_updates)

        self._set_hyper("average_decay", average_decay)
        self._num_updates = num_updates
        self._sequential_update = sequential_update
        self._init = True

    def apply_gradients(self, grads_and_vars, name=None):
        var_list = [v for (_, v) in grads_and_vars]

        if tf.executing_eagerly() and self._init:
            # this to ensure that var_list is registered initially
            self._ema.apply(var_list)
            self._init = False

        train_op = self._optimizer.apply_gradients(grads_and_vars, name=name)

        if self._sequential_update:
            with tf.control_dependencies([train_op]):
                ma_op = self._ema.apply(var_list)
        else:
            ma_op = self._ema.apply(var_list)

        return tf.group(train_op, ma_op, name="train_with_avg")

    def get_config(self):
        config = {
            'average_decay': self._serialize_hyperparameter('average_decay'),
            'num_updates': self._num_updates,
            'sequential_update': self._sequential_update
        }
        base_config = self._optimizer.get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def assign_average_vars(self, var_list):
        """Update variables in var_list with the running mean of the variables.

        Example:
        ```python
        model = tf.Sequential([...])
        opt = tfa.optimizers.MovingAverage(
            tf.keras.optimizers.SGD(lr=2.0), 0.5)

        model.compile(opt, ...)
        model.fit(x, y, ...)

        # Update the weights to their mean before saving
        opt.assign_average_vars(model.variables)

        model.save('model.h5')
        ```
        """
        assign = tf.group([v.assign(self._ema.average(v)) for v in var_list])
        return assign

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
