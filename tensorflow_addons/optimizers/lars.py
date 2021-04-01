# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""LARS optimizer."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked
from typing import Union, Callable, List


@tf.keras.utils.register_keras_serializable(package="Addons")
class LARS(tf.keras.optimizers.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
        - Gradual learning rate warm-up
        - Linear learning rate scaling
        - Poly rule learning rate decay
    Note, LARS scaling is currently only enabled for dense tensors.
    Args:
        lr: A `Tensor` or floating point value. The base learning rate.
        momentum: A floating point value. Momentum hyperparameter.
        weight_decay: A floating point value. Weight decay hyperparameter.
        eeta: LARS coefficient as used in the paper. Dfault set to LARS
            coefficient from the paper. (eeta / weight_decay) determines the
            highest scaling factor in LARS.
        epsilon: Optional epsilon parameter to be set in models that have very
            small gradients. Default set to 0.0.
        nesterov: when set to True, nesterov momentum will be enabled
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        momentum: FloatTensorLike = 0.9,
        weight_decay: FloatTensorLike = 0.0001,
        eeta: FloatTensorLike = 0.001,
        epsilon: FloatTensorLike = 0.0,
        nesterov: bool = False,
        skip_list: List[str] = None,
        name: str = "LARS",
        **kwargs,
    ):

        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay is not positive: %s" % weight_decay)
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("momentum", momentum)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("eeta", eeta)
        self._epsilon = epsilon or tf.keras.backend.epsilon()
        self._nesterov = nesterov
        self._skip_list = skip_list

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "momentum")

    def _compute_lr(self, grad, var, coefficients):
        scaled_lr = coefficients["lr_t"]
        if self._skip_list is None or not any(v in var.name for v in self._skip_list):
            w_norm = tf.linalg.norm(var, ord=2)
            g_norm = tf.linalg.norm(grad, ord=2)
            trust_ratio = tf.where(
                w_norm > 0,
                tf.where(
                    g_norm > 0,
                    (
                        coefficients["eeta_t"]
                        * w_norm
                        / (
                            g_norm
                            + coefficients["weight_decay_t"] * w_norm
                            + self._epsilon
                        )
                    ),
                    1.0,
                ),
                1.0,
            )
            scaled_lr = coefficients["lr_t"] * trust_ratio
            # Add the weight regularization gradient
            grad = grad + coefficients["weight_decay_t"] * var
        return scaled_lr, grad

    def _compute_lr_sparse(self, grad, var, indices, coefficients):
        scaled_lr = coefficients["lr_t"]
        if self._skip_list is None or not any(v in var.name for v in self._skip_list):
            w_norm = tf.linalg.norm(var, ord=2)
            g_norm = tf.linalg.norm(grad, ord=2)
            trust_ratio = tf.where(
                w_norm > 0,
                tf.where(
                    g_norm > 0,
                    (
                        coefficients["eeta_t"]
                        * w_norm
                        / (
                            g_norm
                            + coefficients["weight_decay_t"] * w_norm
                            + self._epsilon
                        )
                    ),
                    1.0,
                ),
                1.0,
            )
            scaled_lr = coefficients["lr_t"] * trust_ratio
            # Add the weight regularization gradient
            var_t = var.assign(
                coefficients["weight_decay_t"] * var, use_locking=self._use_locking
            )
            with tf.control_dependencies([var_t]):
                grad = self._resource_scatter_add(var, indices, grad)
        return scaled_lr, grad

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)].update(
            {
                "momentum_t": tf.identity(self._get_hyper("momentum", var_dtype)),
                "weight_decay_t": tf.identity(
                    self._get_hyper("weight_decay", var_dtype)
                ),
                "eeta_t": tf.identity(self._get_hyper("eeta", var_dtype)),
            }
        )

    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        scaled_lr, grad = self._compute_lr(grad, var, coefficients)
        mom = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceApplyMomentum(
            var=var.handle,
            accum=mom.handle,
            lr=tf.cast(1.0, var.dtype),
            grad=grad * scaled_lr,
            momentum=coefficients["momentum_t"],
            use_locking=self._use_locking,
            use_nesterov=self._nesterov,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        var_device, var_dtype = var.device, var.dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        scaled_lr, grad = self._compute_lr_sparse(grad, var, indices, coefficients)
        grad = tf.IndexedSlices(tf.gather(grad, indices), indices, grad.shape)
        mom = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyMomentum(
            var=var.handle,
            accum=mom.handle,
            lr=tf.cast(1.0, var.dtype),
            grad=grad * scaled_lr,
            indices=indices,
            momentum=coefficients["momentum_t"],
            use_locking=self._use_locking,
            use_nesterov=self._nesterov,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "epsilon": self._epsilon,
                "eeta": self._serialize_hyperparameter("eeta"),
                "nesterov": self._nesterov,
                "skip_list": self._skip_list,
            }
        )
        return config
