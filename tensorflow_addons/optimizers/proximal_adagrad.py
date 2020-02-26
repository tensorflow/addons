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
"""Proximal Adagrad optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class ProximalAdagrad(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Proximal Adagrad algorithm.

    References:

    * [Efficient Learning using Forward-Backward Splitting]
    (http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf).
    """

    def __init__(
        self,
        learning_rate=0.001,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=0.0,
        use_locking=False,
        name="ProximalAdagrad",
        **kwargs
    ):
        """Construct a new Proximal Adagrad optimizer.

        Args:
            learning_rate: A Tensor or a floating point value.
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
            ValueError: If the `initial_accumulator_value`,
                `l1_regularization_strength` or `l2_regularization_strength`
                is invalid.
        """
        if initial_accumulator_value < 0.0:
            raise ValueError(
                "`initial_accumulator_value` must be non-negative: %s"
                % initial_accumulator_value
            )
        if l1_regularization_strength < 0.0:
            raise ValueError(
                "`l1_regularization_strength` must be non-negative: %s"
                % l1_regularization_strength
            )
        if l2_regularization_strength < 0.0:
            raise ValueError(
                "`l2_regularization_strength` must be non-negative: %s"
                % l2_regularization_strength
            )
        super(ProximalAdagrad, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._l1_regularization_strength = l1_regularization_strength
        self._l2_regularization_strength = l2_regularization_strength
        self._initial_accumulator_value = initial_accumulator_value
        self._use_locking = use_locking

    def _create_slots(self, var_list):
        for var in var_list:
            init = tf.keras.initializers.constant(self._initial_accumulator_value)
            self.add_slot(var, "accumulator", init)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(ProximalAdagrad, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
                zero=tf.zeros((), dtype=tf.int64),
            )
        )

    def set_weights(self, weights):
        params = self.weights
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super(ProximalAdagrad, self).set_weights(weights)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Creates an optimizer from its config.

        This method is the reverse of `get_config`, capable of
        instantiating the same optimizer from the config dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.
            custom_objects: A Python dictionary mapping names to additional
                Python objects used to create this optimizer, such as a
                function used for a hyperparameter.

        Returns:
            An optimizer instance.
        """
        if "initial_accumulator_value" not in config:
            config["initial_accumulator_value"] = 0.0
        if "l1_regularization_strength" not in config:
            config["l1_regularization_strength"] = 0.0
        if "l2_regularization_strength" not in config:
            config["l2_regularization_strength"] = 0.0
        if "lr" in config:
            config["learning_rate"] = config.pop("lr")
        return cls(**config)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        acc = self.get_slot(var, "accumulator")
        return training_ops.resource_apply_proximal_adagrad(
            var.handle,
            acc.handle,
            coefficients["lr_t"],
            self._l1_regularization_strength,
            self._l2_regularization_strength,
            grad,
            use_locking=self._use_locking,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        acc = self.get_slot(var, "accumulator")
        return training_ops.resource_sparse_apply_proximal_adagrad(
            var.handle,
            acc.handle,
            coefficients["lr_t"],
            self._l1_regularization_strength,
            self._l2_regularization_strength,
            grad,
            indices,
            use_locking=self._use_locking,
        )

    def get_config(self):
        config = super(ProximalAdagrad, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "l1_regularization_strength": self._l1_regularization_strength,
                "l2_regularization_strength": self._l2_regularization_strength,
                "initial_accumulator_value": self._initial_accumaltor_value,
            }
        )
        return config
