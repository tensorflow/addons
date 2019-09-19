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
        """Construct a new MovingAverage optimizer.

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            average_decay: float. Decay to use to maintain the moving averages
                of trained variables. See `tf.train.ExponentialMovingAverage`
                for details.
            num_updates: Optional count of the number of updates applied to
                variables. See `tf.train.ExponentialMovingAverage` for details.
            sequential_update: Bool. If False, will compute the moving average
                at the same time as the model is updated, potentially doing
                benign data races. If True, will update the moving average
                after gradient updates.
            name: Optional name for the operations created when applying
                gradients. Defaults to "MovingAverage".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super(MovingAverage, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer")

        if num_updates is not None and not isinstance(num_updates, int):
            raise TypeError("num_updates must be None or of integer type")

        if not isinstance(sequential_update, bool):
            raise TypeError("sequential_update must be of bool type")

        with tf.name_scope(name):
            self._ema = tf.train.ExponentialMovingAverage(
                average_decay, num_updates=num_updates)

        self._optimizer = optimizer
        self._set_hyper("average_decay", average_decay)
        self._num_updates = num_updates
        self._sequential_update = sequential_update
        self._initialized = False

    def apply_gradients(self, grads_and_vars, name=None):
        var_list = [v for (_, v) in grads_and_vars]

        if tf.executing_eagerly() and not self._initialized:
            # this to ensure that var_list is registered initially
            self._ema.apply(var_list)
            self._initialized = True

        train_op = self._optimizer.apply_gradients(grads_and_vars, name=name)

        if self._sequential_update:
            with tf.control_dependencies([train_op]):
                ma_op = self._ema.apply(var_list)
        else:
            ma_op = self._ema.apply(var_list)

        return tf.group(train_op, ma_op, name="train_with_avg")

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'average_decay': self._serialize_hyperparameter('average_decay'),
            'num_updates': self._num_updates,
            'sequential_update': self._sequential_update
        }
        base_config = super(MovingAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'), custom_objects=custom_objects)
        return cls(optimizer, **config)

    def assign_average_vars(self, var_list):
        """Assign variables in var_list with their respective moving averages.

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
