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
"""AdaBelief optimizer."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from tensorflow_addons.optimizers import KerasLegacyOptimizer
from typing import Union, Callable, Dict


@tf.keras.utils.register_keras_serializable(package="Addons")
class AdaBelief(KerasLegacyOptimizer):
    """Variant of the Adam optimizer.

    It achieves fast convergence as Adam and generalization comparable to SGD.
    It adapts the step size depending on its "belief" in the gradient direction
    — the optimizer adaptively scales step size by the difference between the
    predicted and observed gradients.

    It implements the AdaBelief proposed by
    Juntang Zhuang et al. in [AdaBelief Optimizer: Adapting stepsizes by the
    belief in observed gradients](https://arxiv.org/abs/2010.07468).

    Example of usage:

    ```python
    opt = tfa.optimizers.AdaBelief(lr=1e-3)
    ```

    Note: `amsgrad` is not described in the original paper. Use it with
          caution.

    You can enable enable warmup by setting `total_steps` and
    `warmup_proportion`,
    and enable recitifcation as in RAdam by setting 'rectify':
    ```python
    opt = tfa.optimizers.AdaBelief(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
        rectify=True,
    )
    ```

    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.

    Note 'rectify' is independent of 'warmup', you can choose any combinations.

    Lookahead, proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1), can be integrated with AdaBelief,
    which is called 'ranger_adabelief' in the author's implementation
    https://github.com/juntang-zhuang/Adabelief-Optimizer.
    The mechanism can be enabled by using the lookahead wrapper. For example:

    ```python
    adabelief = tfa.optimizers.AdaBelief()
    ranger = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)
    ```
    """

    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable, Dict] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-14,
        weight_decay: Union[FloatTensorLike, Callable, Dict] = 0.0,
        amsgrad: bool = False,
        rectify: bool = True,
        sma_threshold: FloatTensorLike = 5.0,
        total_steps: int = 0,
        warmup_proportion: FloatTensorLike = 0.1,
        min_lr: FloatTensorLike = 0.0,
        name: str = "AdaBelief",
        **kwargs,
    ):
        r"""Construct a new RAdam optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                The learning rate.
            beta_1: A float value or a constant float tensor. The exponential
                decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor. The exponential
                decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. Default=1e-14.
                Note that AdaBelief uses epsilon within sqrt (default=1e-14),
                while Adam uses epsilon outside sqrt (default=1e-7).
            weight_decay: A `Tensor` or a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`.
                Weight decay for each parameter.
            amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm
                from the paper "On the Convergence of Adam and beyond".
                sma_threshold. A float value. The threshold for simple mean
                average.
            rectify: boolean. Whether to apply learning rate rectification as
                from RAdam.
            total_steps: An integer. Total number of training steps. Enable
                warmup by setting a value greater than zero.
            warmup_proportion: A floating point value. The proportion of
                increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying
                gradients. Defaults to "RectifiedAdam".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
                `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue`
                is clip gradients by value, `decay` is included for backward
                compatibility to allow time inverse decay of learning rate. `lr`
                is included for backward compatibility, recommended to use
                `learning_rate` instead.
        """
        super().__init__(name, **kwargs)

        if isinstance(learning_rate, Dict):
            learning_rate = tf.keras.optimizers.schedules.deserialize(learning_rate)

        if isinstance(weight_decay, Dict):
            weight_decay = tf.keras.optimizers.schedules.deserialize(weight_decay)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", float(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self.rectify = rectify
        self._has_weight_decay = weight_decay != 0.0
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)
        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)
        return wd_t

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

        m_t = m.assign(
            beta_1_t * m + (1.0 - beta_1_t) * grad,
            use_locking=self._use_locking,
        )
        m_corr_t = m_t / (1.0 - beta_1_power)

        v_t = v.assign(
            beta_2_t * v + (1.0 - beta_2_t) * tf.math.square(grad - m_t) + epsilon_t,
            use_locking=self._use_locking,
        )
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        if self.rectify:
            r_t_numerator = (sma_t - 4.0) * (sma_t - 2.0) * sma_inf
            r_t_denominator = (sma_inf - 4.0) * (sma_inf - 2.0) * sma_t
            r_t = tf.sqrt(r_t_numerator / r_t_denominator)
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(
                sma_t >= sma_threshold,
                r_t * m_corr_t / (v_corr_t + epsilon_t),
                m_corr_t,
            )
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * var

        var_update = var.assign_sub(lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        wd_t = self._decayed_wd(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t, use_locking=self._use_locking)
        m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v = self.get_slot(var, "v")
        m_t_indices = tf.gather(m_t, indices)
        v_scaled_g_values = (
            tf.math.square(grad - m_t_indices) * (1 - beta_2_t) + epsilon_t
        )
        v_t = v.assign(v * beta_2_t, use_locking=self._use_locking)
        v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        if self.rectify:
            r_t_numerator = (sma_t - 4.0) * (sma_t - 2.0) * sma_inf
            r_t_denominator = (sma_inf - 4.0) * (sma_inf - 2.0) * sma_t
            r_t = tf.sqrt(r_t_numerator / r_t_denominator)
            sma_threshold = self._get_hyper("sma_threshold", var_dtype)
            var_t = tf.where(
                sma_t >= sma_threshold,
                r_t * m_corr_t / (v_corr_t + epsilon_t),
                m_corr_t,
            )
        else:
            var_t = m_corr_t / (v_corr_t + epsilon_t)

        if self._has_weight_decay:
            var_t += wd_t * var

        var_update = self._resource_scatter_add(
            var, indices, tf.gather(-lr_t * var_t, indices)
        )

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "decay": self._serialize_hyperparameter("decay"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "rectify": self.rectify,
                "total_steps": int(self._serialize_hyperparameter("total_steps")),
                "warmup_proportion": self._serialize_hyperparameter(
                    "warmup_proportion"
                ),
                "min_lr": self._serialize_hyperparameter("min_lr"),
            }
        )
        return config
