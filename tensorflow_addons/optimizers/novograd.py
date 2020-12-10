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
"""NovoGrad for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typing import Union, Callable
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class NovoGrad(tf.keras.optimizers.Optimizer):
    """Optimizer that implements NovoGrad.

    The NovoGrad Optimizer was first proposed in [Stochastic Gradient
    Methods with Layerwise Adaptive Moments for training of Deep
    Networks](https://arxiv.org/pdf/1905.11286.pdf) NovoGrad is a
    first-order SGD-based algorithm, which computes second moments per
    layer instead of per weight as in Adam. Compared to Adam, NovoGrad
    takes less memory, and has been found to be more numerically stable.
    (For more information on the computation please refer to this
    [link](https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html))

    Second order moment = exponential moving average of Layer-wise square
    of grads:
        v_t <-- beta_2 * v_{t-1} + (1-beta_2) * (g_t)^2
    First order moment in one of four modes:
        1. moment of grads normalized by v_t:
            m_t <- beta_1 * m_{t-1} + [ g_t / (sqrt(v_t)+epsilon)]
        2. moment similar to Adam: exponential moving average of grads
        normalized by v_t (set grad_averaging = True to use this):
            m_t <- beta_1 * m_{t-1} +
                   [(1 - beta_1) * (g_t / (sqrt(v_t) + epsilon))]
        3. weight decay adds a w_d term after grads are rescaled by
        1/sqrt(v_t) (set weight_decay > 0 to use this0:
            m_t <- beta_1 * m_{t-1} +
                   [(g_t / (sqrt(v_t) + epsilon)) + (w_d * w_{t-1})]
        4. weight decay + exponential moving average from Adam:
            m_t <- beta_1 * m_{t-1} +
                   [(1 - beta_1) * ((g_t / (sqrt(v_t + epsilon)) +
                   (w_d * w_{t-1}))]
    Weight update:
        w_t <- w_{t-1} - lr_t * m_t

    Example of usage:
    ```python
    opt = tfa.optimizers.NovoGrad(
        lr=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=0.001,
        grad_averaging=False
    )
    ```
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: FloatTensorLike = 0.0,
        grad_averaging: bool = False,
        amsgrad: bool = False,
        name: str = "NovoGrad",
        **kwargs,
    ):
        r"""Construct a new NovoGrad optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            grad_averaging: determines whether to use Adam style exponential
                moving averaging for the first order moments.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        if weight_decay < 0.0:
            raise ValueError("Weight decay rate cannot be negative")
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("grad_averaging", grad_averaging)
        self.amsgrad = amsgrad
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var=var, slot_name="m", initializer="zeros")
        for var in var_list:
            self.add_slot(
                var=var, slot_name="v", initializer=tf.zeros(shape=[], dtype=var.dtype)
            )
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                one_minus_beta_2_t=1 - beta_2_t,
                one_minus_beta_1_t=1 - beta_1_t,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        weight_decay = self._get_hyper("weight_decay")
        grad_averaging = self._get_hyper("grad_averaging")

        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        v_t = tf.cond(
            tf.equal(self.iterations, 0),
            lambda: g_2,
            lambda: v * coefficients["beta_2_t"]
            + g_2 * coefficients["one_minus_beta_2_t"],
        )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon)
        grad = tf.cond(
            tf.greater(weight_decay, 0), lambda: grad + weight_decay * var, lambda: grad
        )
        grad = tf.cond(
            tf.logical_and(grad_averaging, tf.not_equal(self.iterations, 0)),
            lambda: grad * coefficients["one_minus_beta_1_t"],
            lambda: grad,
        )
        m = self.get_slot(var, "m")
        return tf.raw_ops.ResourceApplyKerasMomentum(
            var=var.handle,
            accum=m.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            momentum=coefficients["beta_1_t"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        weight_decay = self._get_hyper("weight_decay")
        grad_averaging = self._get_hyper("grad_averaging")

        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        # v is just a scalar and does not need to involve sparse tensors.
        v_t = tf.cond(
            tf.equal(self.iterations, 0),
            lambda: g_2,
            lambda: v * coefficients["beta_2_t"]
            + g_2 * coefficients["one_minus_beta_2_t"],
        )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon)
        grad = tf.cond(
            tf.greater(weight_decay, 0),
            lambda: grad + weight_decay * tf.gather(var, indices),
            lambda: grad,
        )
        grad = tf.cond(
            tf.logical_and(grad_averaging, tf.not_equal(self.iterations, 0)),
            lambda: grad * coefficients["one_minus_beta_1_t"],
            lambda: grad,
        )
        m = self.get_slot(var, "m")
        return tf.raw_ops.ResourceSparseApplyKerasMomentum(
            var=var.handle,
            accum=m.handle,
            lr=coefficients["lr_t"],
            grad=grad,
            indices=indices,
            momentum=coefficients["beta_1_t"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "grad_averaging": self._serialize_hyperparameter("grad_averaging"),
            }
        )
        return config
