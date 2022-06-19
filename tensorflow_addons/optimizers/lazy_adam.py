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
"""Variant of the Adam optimizer that handles sparse updates more efficiently.

Compared with the original Adam optimizer, the one in this file can
provide a large improvement in model training throughput for some
applications. However, it provides slightly different semantics than the
original Adam algorithm, and may lead to different empirical results.
"""

import importlib
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked
from typing import Union, Callable


if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
    adam_optimizer_class = tf.keras.optimizers.legacy.Adam
else:
    adam_optimizer_class = tf.keras.optimizers.Adam


@tf.keras.utils.register_keras_serializable(package="Addons")
class LazyAdam(adam_optimizer_class):
    """Variant of the Adam optimizer that handles sparse updates more
    efficiently.

    The original Adam algorithm maintains two moving-average accumulators for
    each trainable variable; the accumulators are updated at every step.
    This class provides lazier handling of gradient updates for sparse
    variables.  It only updates moving-average accumulators for sparse variable
    indices that appear in the current batch, rather than updating the
    accumulators for all indices. Compared with the original Adam optimizer,
    it can provide large improvements in model training throughput for some
    applications. However, it provides slightly different semantics than the
    original Adam algorithm, and may lead to different empirical results.

    Note, amsgrad is currently not supported and the argument can only be
    False.
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        amsgrad: bool = False,
        name: str = "LazyAdam",
        **kwargs,
    ):
        """Constructs a new LazyAdam optimizer.

        Args:
          learning_rate: A `Tensor` or a floating point value. or a schedule
            that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
            The learning rate.
          beta_1: A `float` value or a constant `float` tensor.
            The exponential decay rate for the 1st moment estimates.
          beta_2: A `float` value or a constant `float` tensor.
            The exponential decay rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability.
            This epsilon is "epsilon hat" in
            [Adam: A Method for Stochastic Optimization. Kingma et al., 2014]
            (http://arxiv.org/abs/1412.6980) (in the formula just
            before Section 2.1), not the epsilon in Algorithm 1 of the paper.
          amsgrad: `boolean`. Whether to apply AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and beyond".
            Note that this argument is currently not supported and the
            argument can only be `False`.
          name: Optional name for the operations created when applying
            gradients. Defaults to "LazyAdam".
          **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
            `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue`
            is clip gradients by value, `decay` is included for backward
            compatibility to allow time inverse decay of learning rate. `lr`
            is included for backward compatibility, recommended to use
            `learning_rate` instead.
        """
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        lr = lr_t * tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        # \\(m := beta1 * m + (1 - beta1) * g_t\\)
        m = self.get_slot(var, "m")
        m_t_slice = beta_1_t * tf.gather(m, indices) + (1 - beta_1_t) * grad
        m_update_op = self._resource_scatter_update(m, indices, m_t_slice)

        # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
        v = self.get_slot(var, "v")
        v_t_slice = beta_2_t * tf.gather(v, indices) + (1 - beta_2_t) * tf.math.square(
            grad
        )
        v_update_op = self._resource_scatter_update(v, indices, v_t_slice)

        # \\(variable += -learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
        var_slice = lr * m_t_slice / (tf.math.sqrt(v_t_slice) + epsilon_t)
        var_update_op = self._resource_scatter_sub(var, indices, var_slice)

        return tf.group(*[var_update_op, m_update_op, v_update_op])

    def _resource_scatter_update(self, resource, indices, update):
        return self._resource_scatter_operate(
            resource, indices, update, tf.raw_ops.ResourceScatterUpdate
        )

    def _resource_scatter_sub(self, resource, indices, update):
        return self._resource_scatter_operate(
            resource, indices, update, tf.raw_ops.ResourceScatterSub
        )

    def _resource_scatter_operate(self, resource, indices, update, resource_scatter_op):
        resource_update_kwargs = {
            "resource": resource.handle,
            "indices": indices,
            "updates": update,
        }

        return resource_scatter_op(**resource_update_kwargs)
