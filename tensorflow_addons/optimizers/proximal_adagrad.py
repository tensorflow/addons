# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Proximal Adagrad optimizer."""

from typing import Callable, Union

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.optimizers import KerasLegacyOptimizer
from tensorflow_addons.utils.types import FloatTensorLike


@tf.keras.utils.register_keras_serializable(package="Addons")
class ProximalAdagrad(KerasLegacyOptimizer):
    """Optimizer that implements the Proximal Adagrad algorithm.

    References:
        - [Efficient Learning using Forward-Backward Splitting](
          http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf).
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        initial_accumulator_value: float = 0.1,
        l1_regularization_strength: float = 0.0,
        l2_regularization_strength: float = 0.0,
        name: str = "ProximalAdagrad",
        **kwargs,
    ):
        """Construct a new Proximal Adagrad optimizer.

        Args:
            learning_rate: A Tensor or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                The learning rate.
            initial_accumulator_value: A floating point value.
                Starting value for the accumulators, must be positive.
            l1_regularization_strength: A floating point value.
                The l1 regularization term, must be greater than or
                equal to zero.
            l2_regularization_strength: A floating point value.
                The l2 regularization term, must be greater than or
                equal to zero.
            name: Optional name for the operations created when applying
                gradients. Defaults to "ProximalAdagrad".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        Raises:
            ValueError: If the `initial_accumulator_value`, `l1` or `l2`
                is invalid.
        """
        if initial_accumulator_value < 0.0:
            raise ValueError("`initial_accumulator_value` must be non-negative.")
        if l1_regularization_strength < 0.0:
            raise ValueError("`l1_regularization_strength` must be non-negative.")
        if l2_regularization_strength < 0.0:
            raise ValueError("`l2_regularization_strength` must be non-negative.")
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("l1_regularization_strength", l1_regularization_strength)
        self._set_hyper("l2_regularization_strength", l2_regularization_strength)
        self._initial_accumulator_value = initial_accumulator_value

    def _create_slots(self, var_list):
        for var in var_list:
            init = tf.keras.initializers.constant(self._initial_accumulator_value)
            self.add_slot(var, "accumulator", init)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        acc = self.get_slot(var, "accumulator")
        return tf.raw_ops.ResourceApplyProximalAdagrad(
            var=var.handle,
            accum=acc.handle,
            lr=coefficients["lr_t"],
            l1=coefficients["l1_regularization_strength"],
            l2=coefficients["l2_regularization_strength"],
            grad=grad,
            use_locking=self._use_locking,
        )

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(
            {
                "l1_regularization_strength": tf.identity(
                    self._get_hyper("l1_regularization_strength", var_dtype)
                ),
                "l2_regularization_strength": tf.identity(
                    self._get_hyper("l2_regularization_strength", var_dtype)
                ),
            }
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        acc = self.get_slot(var, "accumulator")
        return tf.raw_ops.ResourceSparseApplyProximalAdagrad(
            var=var.handle,
            accum=acc.handle,
            lr=coefficients["lr_t"],
            l1=coefficients["l1_regularization_strength"],
            l2=coefficients["l2_regularization_strength"],
            grad=grad,
            indices=indices,
            use_locking=self._use_locking,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "initial_accumulator_value": self._initial_accumulator_value,
                "l1_regularization_strength": self._serialize_hyperparameter(
                    "l1_regularization_strength"
                ),
                "l2_regularization_strength": self._serialize_hyperparameter(
                    "l2_regularization_strength"
                ),
            }
        )
        return config
