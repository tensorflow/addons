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
import tensorflow as tf
from tensorflow_addons.utils import types
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class GradientAccumulator(tf.keras.optimizers.Optimizer):
    """Optimizer wrapper for gradient accumulation."""

    @typechecked
    def __init__(
        self,
        inner_optimizer: types.Optimizer,
        accum_steps: types.TensorLike = 4,
        name: str = "GradientAccumulator",
        **kwargs,
    ):
        r"""Construct a new GradientAccumulator optimizer.

        Args:
            inner_optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulator".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._optimizer = tf.keras.optimizers.get(inner_optimizer)
        self._step = None
        self._gradients = {}
        self._accum_steps = accum_steps

        def _accum_grad(grads_and_vars):
            with tf.init_scope():
                if not self._gradients:
                    for grad, var in grads_and_vars:
                        if tf.distribute.has_strategy():
                            for v in var.values:
                                self._gradients[v.ref()] = tf.Variable(
                                    tf.zeros_like(v), trainable=False
                                )
                        else:
                            self._gradients[var.ref()] = tf.Variable(
                                tf.zeros_like(var), trainable=False
                            )
            new_grads_and_vars = []
            for grad, var in grads_and_vars:
                if tf.distribute.has_strategy():
                    replica_id = tf.get_static_value(
                        tf.distribute.get_replica_context().replica_id_in_sync_group
                    )
                    handle = self._gradients[var.values[replica_id].ref()]
                else:
                    handle = self._gradients[var.ref()]

                if isinstance(grad, tf.IndexedSlices):
                    handle.scatter_add(grad)
                    fake_grad = tf.IndexedSlices(
                        tf.zeros_like(grad.values), grad.indices, grad.dense_shape
                    )
                else:
                    handle.assign_add(grad)
                    fake_grad = tf.zeros_like(var)

                def _get_grad():
                    new_grad = handle.read_value()
                    handle.assign(tf.zeros_like(handle), use_locking=self._use_locking)
                    return new_grad

                new_grad = tf.cond(
                    self.step % self._accum_steps == 0,
                    _get_grad,
                    lambda: fake_grad,
                )
                new_grads_and_vars.append((new_grad, var))
            return new_grads_and_vars

        self.gradient_transformers.append(_accum_grad)
        self._iterations = self._optimizer.iterations

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)

    @property
    def step(self):
        """Variable. The number of training steps this Optimizer has run."""
        if self._step is None:
            with self._distribution_strategy_scope():
                self._step = self.add_weight(
                    "iter",
                    shape=[],
                    initializer="ones",
                    dtype=tf.int64,
                    trainable=False,
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                )
            self._weights.append(self._step)
        return self._step

    @step.setter
    def step(self, variable):
        if self._step is not None:
            raise RuntimeError(
                "Cannot set `step` to a new Variable after "
                "the Optimizer weights have been created"
            )
        self._step = variable
        self._weights.append(self._step)

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            raise ValueError(
                "The accumulator should be called first to initialize the gradients"
            )
        return list(
            gradient.read_value() if gradient is not None else gradient
            for _, gradient in self._gradients
        )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        train_op = super().apply_gradients(grads_and_vars, name, **kwargs)
        with tf.control_dependencies([train_op]):
            with tf.control_dependencies(
                [
                    self._optimizer.iterations.assign_add(
                        tf.cast(
                            tf.where(self.step % self._accum_steps == 0, 1, 0), tf.int64
                        ),
                        read_value=False,
                    )
                ]
            ):
                return self.step.assign_add(1, read_value=False)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        if "apply_state" in self._optimizer._dense_apply_args:
            train_op = self._optimizer._resource_apply_dense(
                grad,
                var,
                apply_state=apply_state,
            )
        else:
            train_op = self._optimizer._resource_apply_dense(grad, var)
        return train_op

    def _resource_apply_sparse(self, grad: types.TensorLike, var, indices, apply_state):
        if "apply_state" in self._optimizer._sparse_apply_args:
            train_op = self._optimizer._resource_apply_sparse(
                grad, var, indices, apply_state=apply_state
            )
        else:
            train_op = self._optimizer._resource_apply_sparse(grad, var, indices)
        return train_op

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for _, gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    )
                )

        return tf.group(assign_ops)

    @property
    def inner_optimizer(self):
        """The optimizer that this LossScaleOptimizer is wrapping."""
        return self._optimizer

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

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

    def get_config(self):
        config = {
            "accum_steps": self._accum_steps,
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
