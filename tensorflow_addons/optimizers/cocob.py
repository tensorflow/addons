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
"""COntinuos COin Betting (COCOB) Backprop optimizer"""

from typeguard import typechecked
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Addons")
class COCOB(tf.keras.optimizers.Optimizer):
    """Optimizer that implements COCOB Backprop Algorithm

        Reference:
            - [COntinuos COin Betting (COCOB) Backprop optimizer
    ](https://arxiv.org/abs/1705.07795)
    """

    @typechecked
    def __init__(
        self,
        alpha: float = 100,
        use_locking: bool = False,
        name: str = "COCOB",
        **kwargs,
    ):
        """Constructs a new COCOB-Backprop optimizer

        Arguments:
            `aplha`: Default value is set to 100 as per paper.
                     This has the effect of restricting the value of the
                     parameters in the first iterations of the algorithm.
                     (Refer to Paper for indepth understanding)

        Rasies:
            `ValueError`: If the value of `alpha` is less than 1.
            `NotImplementedError`: If the data is in sparse format.
        """

        if alpha < 1:
            raise ValueError("`alpha` must be greater than Zero")

        super().__init__(name, **kwargs)
        self._set_hyper("alpha", alpha)
        self._alpha = alpha

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "lr", initializer=tf.keras.initializers.Constant(1e-8))
            self.add_slot(v, "grad_norm_sum")
            self.add_slot(v, "gradients_sum")
            self.add_slot(v, "tilde_w")
            self.add_slot(v, "reward")

    def _resource_apply_dense(self, grad, handle, apply_state=None):
        gradients_sum = self.get_slot(handle, "gradients_sum")
        grad_norm_sum = self.get_slot(handle, "grad_norm_sum")
        tilde_w = self.get_slot(handle, "tilde_w")
        lr = self.get_slot(handle, "lr")
        reward = self.get_slot(handle, "reward")

        lr_update = tf.maximum(lr, tf.abs(grad))
        gradients_sum_update = gradients_sum + grad
        grad_norm_sum_update = grad_norm_sum + tf.abs(grad)
        reward_update = tf.maximum(reward - grad * tilde_w, 0)

        grad_max = tf.maximum(grad_norm_sum_update + lr_update, self._alpha * lr_update)
        rewards_lr_sum = reward_update + lr_update
        new_w = -gradients_sum_update / (lr_update * (grad_max)) * rewards_lr_sum

        var_update = handle - tilde_w + new_w
        tilde_w_update = new_w

        gradients_sum_update_op = gradients_sum.assign(gradients_sum_update)
        grad_norm_sum_update_op = grad_norm_sum.assign(grad_norm_sum_update)
        var_update_op = handle.assign(var_update)
        tilde_w_update_op = tilde_w.assign(tilde_w_update)
        lr_update_op = lr.assign(lr_update)
        reward_update_op = reward.assign(reward_update)

        return tf.group(
            *[
                gradients_sum_update_op,
                var_update_op,
                grad_norm_sum_update_op,
                tilde_w_update_op,
                reward_update_op,
                lr_update_op,
            ]
        )

    def _resource_apply_sparse(self, grad, handle, indices, apply_state=None):
        raise NotImplementedError()

    def get_config(self):

        config = {
            "alpha": self._serialize_hyperparameter("alpha"),
        }
        base_config = super().get_config()
        return {**base_config, **config}
