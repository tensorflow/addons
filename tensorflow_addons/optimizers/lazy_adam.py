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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class LazyAdam(tf.keras.optimizers.Adam):
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

    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 amsgrad=False,
                 name='LazyAdam',
                 **kwargs):
        super(LazyAdam, self).__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.math.pow(beta_1_t, local_step)
        beta_2_power = tf.math.pow(beta_2_t, local_step)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        lr = (lr_t * tf.math.sqrt(1 - beta_2_power) / (1 - beta_1_power))

        # \\(m := beta1 * m + (1 - beta1) * g_t\\)
        m = self.get_slot(var, "m")
        m_t_slice = beta_1_t * tf.gather(m, indices) + (1 - beta_1_t) * grad

        m_update_kwargs = {
            'resource': m.handle,
            'indices': indices,
            'updates': m_t_slice
        }
        m_update_op = tf.raw_ops.ResourceScatterUpdate(**m_update_kwargs)

        # \\(v := beta2 * v + (1 - beta2) * (g_t * g_t)\\)
        v = self.get_slot(var, "v")
        v_t_slice = (beta_2_t * tf.gather(v, indices) +
                     (1 - beta_2_t) * tf.math.square(grad))

        v_update_kwargs = {
            'resource': v.handle,
            'indices': indices,
            'updates': v_t_slice
        }
        v_update_op = tf.raw_ops.ResourceScatterUpdate(**v_update_kwargs)

        # \\(variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))\\)
        var_slice = lr * m_t_slice / (tf.math.sqrt(v_t_slice) + epsilon_t)

        var_update_kwargs = {
            'resource': var.handle,
            'indices': indices,
            'updates': var_slice
        }
        var_update_op = tf.raw_ops.ResourceScatterSub(**var_update_kwargs)

        return tf.group(*[var_update_op, m_update_op, v_update_op])
