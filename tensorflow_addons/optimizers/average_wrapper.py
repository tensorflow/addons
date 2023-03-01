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

import abc
import warnings

import tensorflow as tf
from tensorflow_addons.optimizers import KerasLegacyOptimizer
from tensorflow_addons.utils import types
from typeguard import typechecked


class AveragedOptimizerWrapper(KerasLegacyOptimizer, metaclass=abc.ABCMeta):
    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        name: str = "AverageOptimizer",
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        if isinstance(optimizer, str):
            if (
                hasattr(tf.keras.optimizers, "legacy")
                and KerasLegacyOptimizer == tf.keras.optimizers.legacy.Optimizer
            ):
                optimizer = tf.keras.optimizers.get(
                    optimizer, use_legacy_optimizer=True
                )
            else:
                optimizer = tf.keras.optimizers.get(optimizer)

        if not isinstance(optimizer, KerasLegacyOptimizer):
            raise TypeError(
                "optimizer is not an object of tf.keras.optimizers.legacy.Optimizer "
            )

        self._optimizer = optimizer
        self._track_trackable(self._optimizer, "awg_optimizer")

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "average")

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare_local(self, var_device, var_dtype, apply_state):
        return self._optimizer._prepare_local(var_device, var_dtype, apply_state)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    @abc.abstractmethod
    def average_op(self, var, average_var, local_apply_state):
        raise NotImplementedError

    def _apply_average_op(self, train_op, var, apply_state):
        apply_state = apply_state or {}
        local_apply_state = apply_state.get((var.device, var.dtype.base_dtype))
        if local_apply_state is None:
            local_apply_state = self._fallback_apply_state(
                var.device, var.dtype.base_dtype
            )
        average_var = self.get_slot(var, "average")
        return self.average_op(var, average_var, local_apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if "apply_state" in self._optimizer._dense_apply_args:
            train_op = self._optimizer._resource_apply_dense(
                grad, var, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_dense(grad, var)
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        average_op = self._apply_average_op(train_op, var, apply_state)
        return tf.group(train_op, average_op)

    def _resource_apply_sparse_duplicate_indices(
        self, grad, var, indices, apply_state=None
    ):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse_duplicate_indices(
                grad, var, indices
            )
        average_op = self._apply_average_op(train_op, var, apply_state)
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
        assign_ops = []
        for var in var_list:
            try:
                assign_ops.append(
                    var.assign(
                        self.get_slot(var, "average"),
                        use_locking=self._use_locking,
                    )
                )
            except Exception as e:
                warnings.warn("Unable to assign average slot to {} : {}".format(var, e))
        return tf.group(assign_ops)

    def get_config(self):
        config = {
            "optimizer": tf.keras.optimizers.serialize(self._optimizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop("optimizer"), custom_objects=custom_objects
        )
        return cls(optimizer, **config)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)  #

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)
