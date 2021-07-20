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
from typing import Union, Dict, Hashable, List

import tensorflow as tf

from tensorflow_addons.utils import types


class AccumulationGradientTransformer:
    _accu_gradients: Union[Dict[Hashable, tf.Variable], None] = None

    def __init__(
        self,
        optimizer: types.Optimizer,
        accu_steps: types.TensorLike,
        trainable_variables,
    ):
        self.optimizer = optimizer
        self.accu_steps = accu_steps
        self.step = tf.Variable(0, dtype=tf.int64, name="ga_step")
        self._accu_gradients: Union[List[tf.Variable], None] = None
        policy = tf.keras.mixed_precision.global_policy()
        self.variable_dtype = policy.variable_dtype
        self._accu_gradients = {
            var.ref(): self.optimizer.add_slot(var, "ga") for var in trainable_variables
        }

    def __call__(self, grads_and_vars, *args, **kwargs):

        variables = [var for (_, var) in grads_and_vars]
        accu_gradients = self._accu_gradients
        step_inc_op = self.step.assign_add(1, read_value=False)

        with tf.control_dependencies([step_inc_op]):
            can_apply = tf.cast(
                self.step % self.accu_steps == 0, dtype=self.variable_dtype
            )
            accumulate = tf.cast(
                self.step % (self.accu_steps + 1) != 0, dtype=self.variable_dtype
            )

        accum_ops = list()
        for grad, var in grads_and_vars:

            # Get the accumulated gradient
            grad_accum = accu_gradients[var.ref()] * accumulate

            if isinstance(grad, tf.IndexedSlices):
                added = tf.IndexedSlices(
                    values=grad.values + tf.gather_nd(grad_accum, grad.indices),
                    indices=grad.indices,
                    dense_shape=grad.dense_shape,
                )
                accu_op = accu_gradients[var.ref()].scatter_update(added)
            else:
                accu_op = accu_gradients[var.ref()].assign(
                    grad + grad_accum, read_value=False
                )

            accum_ops.append(accu_op)

        iter_dec_op = self.optimizer.iterations.assign_add(
            -1 * tf.cast(can_apply, dtype=self.optimizer.iterations.dtype),
            read_value=False,
        )

        with tf.control_dependencies(accum_ops + [iter_dec_op]):
            gradients = [accu_gradients[var.ref()] * can_apply for var in variables]
            return list(zip(gradients, variables))


def GradientAccumulator(
    optimizer: types.Optimizer, accu_steps: int = 2, trainable_variables=None
) -> types.Optimizer:
    if trainable_variables is None:
        trainable_variables = list()

    if isinstance(optimizer, str):
        optimizer = tf.keras.optimizers.get(optimizer)

    optimizer.gradient_transformers.append(
        AccumulationGradientTransformer(
            optimizer=optimizer,
            accu_steps=accu_steps,
            trainable_variables=trainable_variables,
        )
    )

    return optimizer


# class GradientAccumulator(tf.keras.optimizers.Optimizer):
#     """Optimizer wrapper for gradient accumulation."""
#
#     @typechecked
#     def __init__(
#         self,
#         inner_optimizer: types.Optimizer,
#         accum_steps: types.TensorLike = 4,
#         reduction: str = "SUM",
#         name: str = "GradientAccumulator",
#         **kwargs,
#     ):
#         r"""Construct a new GradientAccumulator optimizer.
#
#         Args:
#             inner_optimizer: str or `tf.keras.optimizers.Optimizer` that will be
#                 used to compute and apply gradients.
#             accum_steps: int > 0. Update gradient in every accumulation steps.
#             reduction: str, Reduction method ['SUM', 'MEAN']
#             name: Optional name for the operations created when applying
#                 gradients. Defaults to "GradientAccumulator".
#             **kwargs: keyword arguments. Allowed to be {`clipnorm`,
#                 `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
#                 norm; `clipvalue` is clip gradients by value, `decay` is
#                 included for backward compatibility to allow time inverse
#                 decay of learning rate. `lr` is included for backward
#                 compatibility, recommended to use `learning_rate` instead.
#         """
#         super().__init__(name, **kwargs)
#         self._optimizer = tf.keras.optimizers.get(inner_optimizer)
#         self._step = None
#         self._gradients = {}
#         self._accum_steps = accum_steps
#         self._reduction = reduction
#
#         def _accum_grad(grads_and_vars):
#             with tf.init_scope():
#                 if not self._gradients:
#                     for grad, var in grads_and_vars:
#                         if tf.distribute.has_strategy():
#                             for v in var.values:
#                                 self._gradients[v.ref()] = tf.Variable(
#                                     tf.zeros_like(v), trainable=False
#                                 )
#                         else:
#                             self._gradients[var.ref()] = tf.Variable(
#                                 tf.zeros_like(var), trainable=False
#                             )
#             new_grads_and_vars = []
#             for grad, var in grads_and_vars:
#                 if tf.distribute.has_strategy():
#                     replica_id = tf.get_static_value(
#                         tf.distribute.get_replica_context().replica_id_in_sync_group
#                     )
#                     handle = self._gradients[var.values[replica_id].ref()]
#                 else:
#                     handle = self._gradients[var.ref()]
#
#                 if isinstance(grad, tf.IndexedSlices):
#                     handle.scatter_add(grad)
#
#                     def _get_grad():
#                         new_grad = handle.read_value()
#                         if self._reduction == "MEAN":
#                             new_grad /= tf.cast(self._accum_steps, new_grad.dtype)
#                         indices = tf.squeeze(
#                             tf.where(
#                                 tf.reduce_sum(
#                                     new_grad, axis=list(range(len(new_grad.shape))[1:])
#                                 )
#                                 != 0
#                             ),
#                             axis=-1,
#                         )
#
#                         values = tf.gather(new_grad, indices)
#                         dense_shape = tf.constant(new_grad.shape.as_list())
#                         handle.assign(
#                             tf.zeros_like(handle), use_locking=self._use_locking
#                         )
#                         return values, tf.cast(indices, tf.int32), dense_shape
#
#                     values, indices, dense_shape = tf.cond(
#                         self.step % self._accum_steps == 0,
#                         _get_grad,
#                         lambda: (
#                             tf.zeros_like(grad.values),
#                             grad.indices,
#                             grad.dense_shape,
#                         ),
#                     )
#                     new_grad = tf.IndexedSlices(values, indices, dense_shape)
#                     new_grads_and_vars.append((new_grad, var))
#                 else:
#                     handle.assign_add(grad)
#
#                     def _get_grad():
#                         new_grad = handle.read_value()
#                         if self._reduction == "MEAN":
#                             new_grad /= tf.cast(self._accum_steps, new_grad.dtype)
#                         handle.assign(
#                             tf.zeros_like(handle), use_locking=self._use_locking
#                         )
#                         return new_grad
#
#                     new_grad = tf.cond(
#                         self.step % self._accum_steps == 0,
#                         _get_grad,
#                         lambda: tf.zeros_like(grad),
#                     )
#                     new_grads_and_vars.append((new_grad, var))
#             return new_grads_and_vars
#
#         self._optimizer.gradient_transformers.append(_accum_grad)
#         self._iterations = self._optimizer.iterations
#
#     def _create_slots(self, var_list):
#         self._optimizer._create_slots(var_list=var_list)
#
#     @property
#     def step(self):
#         """Variable. The number of training steps this Optimizer has run."""
#         if self._step is None:
#             with self._distribution_strategy_scope():
#                 self._step = self.add_weight(
#                     "iter",
#                     shape=[],
#                     initializer="ones",
#                     dtype=tf.int64,
#                     trainable=False,
#                     aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
#                 )
#             self._weights.append(self._step)
#         return self._step
#
#     @step.setter
#     def step(self, variable):
#         if self._step is not None:
#             raise RuntimeError(
#                 "Cannot set `step` to a new Variable after "
#                 "the Optimizer weights have been created"
#             )
#         self._step = variable
#         self._weights.append(self._step)
#
#     @property
#     def gradients(self):
#         """The accumulated gradients on the current replica."""
#         if not self._gradients:
#             raise ValueError(
#                 "The accumulator should be called first to initialize the gradients"
#             )
#         return list(
#             gradient.read_value() if gradient is not None else gradient
#             for _, gradient in self._gradients
#         )
#
#     def apply_gradients(self, grads_and_vars, name=None, **kwargs):
#         train_op = self._optimizer.apply_gradients(grads_and_vars, name, **kwargs)
#         with tf.control_dependencies([train_op]):
#             with tf.control_dependencies(
#                 [
#                     self._optimizer.iterations.assign_add(
#                         tf.cast(self.step % self._accum_steps == 0, tf.int64),
#                         read_value=False,
#                     )
#                 ]
#             ):
#                 return self.step.assign_add(1, read_value=False)
#
#     def reset(self):
#         """Resets the accumulated gradients on the current replica."""
#         assign_ops = []
#         if not self._gradients:
#             return assign_ops
#
#         for _, gradient in self._gradients:
#             if gradient is not None:
#                 assign_ops.append(
#                     gradient.assign(
#                         tf.zeros_like(gradient),
#                         use_locking=self._use_locking,
#                         read_value=False,
#                     )
#                 )
#
#         return tf.group(assign_ops)
#
#     @property
#     def inner_optimizer(self):
#         """The optimizer that this LossScaleOptimizer is wrapping."""
#         return self._optimizer
#
#     @property
#     def iterations(self):
#         return self._optimizer.iterations
#
#     @iterations.setter
#     def iterations(self, variable):
#         self._optimizer.iterations = variable
#
#     @property
#     def lr(self):
#         return self._optimizer._get_hyper("learning_rate")
#
#     @lr.setter
#     def lr(self, lr):
#         self._optimizer._set_hyper("learning_rate", lr)  #
#
#     @property
#     def learning_rate(self):
#         return self._optimizer._get_hyper("learning_rate")
#
#     @learning_rate.setter
#     def learning_rate(self, learning_rate):
#         self._optimizer._set_hyper("learning_rate", learning_rate)
#
#     def get_config(self):
#         config = {
#             "accum_steps": self._accum_steps,
#             "optimizer": tf.keras.optimizers.serialize(self._optimizer),
#         }
#         base_config = super().get_config()
#         return {**base_config, **config}
#
#     @classmethod
#     def from_config(cls, config, custom_objects=None):
#         optimizer = tf.keras.optimizers.deserialize(
#             config.pop("optimizer"), custom_objects=custom_objects
#         )
#         return cls(optimizer, **config)
