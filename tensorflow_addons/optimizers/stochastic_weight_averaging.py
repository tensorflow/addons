# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An implementation of the Stochastic Weight Averaging optimizer.

The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
et. al in the paper [Averaging Weights Leads to Wider Optima and Better
Generalization](https://arxiv.org/abs/1803.05407). The optimizer
implements averaging of multiple points along the trajectory of SGD.
This averaging has shown to improve model performance on validation/test
sets whilst possibly causing a small increase in loss on the training
set.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Addons')
class SWA(tf.keras.optimizers.Optimizer):
    """This class extends optimizers with Stochastic Weight Averaging (SWA).

    The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
    et. al in the paper [Averaging Weights Leads to Wider Optima and
    Better Generalization](https://arxiv.org/abs/1803.05407). The optimizer
    implements averaging of multiple points along the trajectory of SGD. The
    optimizer expects an inner optimizer which will be used to apply the
    gradients to the variables and itself computes a running average of the
    variables every `k` steps (which generally corresponds to the end
    of a cycle when a cyclic learning rate is employed).

    We also allow the specification of the number of steps averaging
    should first happen after. Let's say, we want averaging to happen every `k`
    steps after the first `m` steps. After step `m` we'd take a snapshot of the
    variables and then average the weights appropriately at step `m + k`,
    `m + 2k` and so on. The assign_average_vars function can be called at the
    end of training to obtain the averaged_weights from the optimizer.

    Note: If your model has batch-normalization layers you would need to run
    the final weights through the data to compute the running mean and
    variance corresponding to the activations for each layer of the network.
    From the paper: If the DNN uses batch normalization we run one
    additional pass over the data, to compute the running mean and standard
    deviation of the activations for each layer of the network with SWA
    weights after the training is finished, since these statistics are not
    collected during training. For most deep learning libraries, such as
    PyTorch or Tensorflow, one can typically collect these statistics by
    making a forward pass over the data in training mode
    ([Averaging Weights Leads to Wider Optima and Better
    Generalization](https://arxiv.org/abs/1803.05407))

    Example of usage:

    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    ```
    """

    def __init__(self,
                 optimizer,
                 start_averaging=0,
                 average_period=10,
                 name='SWA',
                 **kwargs):
        r"""Wrap optimizer with the Stochastic Weight Averaging mechanism.

        Args:
            optimizer: The original optimizer that will be used to compute and
              apply the gradients.
            start_averaging: An integer. Threshold to start averaging using 
              SWA. Averaging only occurs at `start_averaging` iterations, must
              be >= 0. If start_averaging = m, the first snapshot will be 
              taken after the mth application of gradients (where the first
              iteration is iteration 0).
            average_period: An integer. The synchronization period of SWA. The
              averaging occurs every average_period steps. Averaging period
              needs to be >= 1.
            name: Optional name for the operations created when applying
              gradients. Defaults to 'SWA'.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, 
              `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by 
              norm; `clipvalue` is clip gradients by value, `decay` is 
              included for backward compatibility to allow time inverse 
              decay of learning rate. `lr` is included for backward 
              compatibility, recommended to use `learning_rate` instead.
        """
        super(SWA, self).__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                'optimizer is not an object of tf.keras.optimizers.Optimizer')
        if average_period < 1:
            raise ValueError('average_period must be >= 1')
        if start_averaging < 0:
            raise ValueError('start_averaging must be >= 0')

        self._optimizer = optimizer
        self._set_hyper('average_period', average_period)
        self._set_hyper('start_averaging', start_averaging)
        self._initialized = False

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, 'average')

    def _create_hypers(self):
        self._optimizer._create_hypers()  # pylint: disable=protected-access

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations  # pylint: disable=protected-access
        return super(SWA, self).apply_gradients(grads_and_vars, name)

    def _average_op(self, var):
        average_var = self.get_slot(var, 'average')
        average_period = self._get_hyper('average_period', tf.dtypes.int64)
        start_averaging = self._get_hyper('start_averaging', tf.dtypes.int64)
        # check if the correct number of iterations has taken place to start
        # averaging.
        thresold_cond = tf.greater_equal(self.iterations, start_averaging)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging,
                             average_period))
        # checks if the iteration is one in which a snapshot should be taken.
        sync_cond = tf.equal(start_averaging + num_snapshots * average_period,
                             self.iterations)
        num_snapshots = tf.cast(num_snapshots, tf.float32)
        average_value = (
            (average_var * num_snapshots + var) / (num_snapshots + 1.))
        average_cond = tf.reduce_all([thresold_cond, sync_cond])
        with tf.control_dependencies([average_value]):
            average_update = average_var.assign(
                tf.where(
                    average_cond,
                    average_value,
                    average_var,
                ),
                use_locking=self._use_locking)
        return average_update

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def _resource_apply_dense(self, grad, var):
        train_op = self._optimizer._resource_apply_dense(grad, var)  # pylint: disable=protected-access
        with tf.control_dependencies([train_op]):
            average_op = self._average_op(var)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices):
        train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
            grad, var, indices)
        with tf.control_dependencies([train_op]):
            average_op = self._average_op(var)
        return tf.group(train_op, average_op)

    def assign_average_vars(self, var_list):
        """Assign variables in var_list with their respective averages.

        Args:
            var_list: List of model variables to be assigned to their average.

        Returns:
            assign_op: The op corresponding to the assignment operation of
            variables to their average.

        Example:
        ```python
        model = tf.Sequential([...])
        opt = tfa.optimizers.SWA(
                tf.keras.optimizers.SGD(lr=2.0), 100, 10)
        model.compile(opt, ...)
        model.fit(x, y, ...)

        # Update the weights to their mean before saving
        opt.assign_average_vars(model.variables)

        model.save('model.h5')
        ```
        """
        assign_op = tf.group(
            [var.assign(self.get_slot(var, 'average')) for var in var_list])
        return assign_op

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'average_period': self._serialize_hyperparameter('average_period'),
            'start_averaging':
            self._serialize_hyperparameter('start_averaging')
        }
        base_config = super(SWA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def lr(self):
        return self._optimizer._get_hyper('learning_rate')  # pylint: disable=protected-access

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper('learning_rate', lr)  # pylint: disable=protected-access

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper('learning_rate')  # pylint: disable=protected-access

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper('learning_rate', learning_rate)  # pylint: disable=protected-access

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)
