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

import tensorflow as tf
from tensorflow_addons.utils import types

from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
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

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        sync_period: int = 6,
        slow_step_size: types.FloatTensorLike = 0.5,
        name: str = "Lookahead",
        **kwargs,
    ):
        r"""Wrap optimizer with the lookahead mechanism.

        Args:
            optimizer: The original optimizer that will be used to compute
                and apply the gradients.
            sync_period: An integer. The synchronization period of lookahead.
                Enable lookahead mechanism by setting it with a positive value.
            slow_step_size: A floating point value.
                The ratio for updating the slow weights.
            name: Optional name for the operations created when applying
                gradients. Defaults to "Lookahead".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.Optimizer"
            )

        self._optimizer = optimizer
        self._set_hyper("sync_period", sync_period)
        self._set_hyper("slow_step_size", slow_step_size)
        self._initialized = False
        self._track_trackable(self._optimizer, "lh_base_optimizer")

    def _create_slots(self, var_list):
        self._optimizer._create_slots(
            var_list=var_list
        )  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, "slow")

    def _create_hypers(self):
        self._optimizer._create_hypers()  # pylint: disable=protected-access

    def _prepare(self, var_list):
        return self._optimizer._prepare(
            var_list=var_list
        )  # pylint: disable=protected-access

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = (
            self.iterations
        )  # pylint: disable=protected-access
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    def _init_op(self, var):
        slow_var = self.get_slot(var, "slow")
        return slow_var.assign(
            tf.where(
                tf.equal(self.iterations, tf.constant(0, dtype=self.iterations.dtype)),
                var,
                slow_var,
            ),
            use_locking=self._use_locking,
        )

    def _look_ahead_op(self, var):
        var_dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, "slow")
        local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
        sync_period = self._get_hyper("sync_period", tf.dtypes.int64)
        slow_step_size = self._get_hyper("slow_step_size", var_dtype)
        step_back = slow_var + slow_step_size * (var - slow_var)
        sync_cond = tf.equal(
            tf.math.floordiv(local_step, sync_period) * sync_period, local_step
        )
        with tf.control_dependencies([step_back]):
            slow_update = slow_var.assign(
                tf.where(sync_cond, step_back, slow_var), use_locking=self._use_locking
            )
            var_update = var.assign(
                tf.where(sync_cond, step_back, var), use_locking=self._use_locking
            )
        return tf.group(slow_update, var_update)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def _resource_apply_dense(self, grad, var):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_dense(
                grad, var
            )  # pylint: disable=protected-access
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def _resource_apply_sparse(self, grad, var, indices):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
                grad, var, indices
            )
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
            "sync_period": self._serialize_hyperparameter("sync_period"),
            "slow_step_size": self._serialize_hyperparameter("slow_step_size"),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, lr):
        self.learning_rate = lr

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)
