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
"""Conditional Gradient optimizer."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked
from typing import Union, Callable


@tf.keras.utils.register_keras_serializable(package="Addons")
class ConditionalGradient(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Conditional Gradient optimization.

    This optimizer helps handle constraints well.

    Currently only supports frobenius norm constraint or nuclear norm
    constraint.
    See https://arxiv.org/pdf/1803.06453.pdf

    ```
    variable -= (1-learning_rate) * (variable + lambda_ * gradient
        / (frobenius_norm(gradient) + epsilon))
    ```

    Note that `lambda_` here refers to the constraint "lambda" in
    the paper. `epsilon` is constant with tiny value as compared to
    the value of frobenius norm of gradient. The purpose of `epsilon`
    here is to avoid the case that the value of frobenius norm of
    gradient is 0.

    In this implementation, `epsilon` defaults to $10^{-7}$.

    For nucler norm constraint, the formula is as following:

    ```
    variable -= (1-learning_rate) * (variable
        + lambda_ * top_singular_vector(gradient))
    ```
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable],
        lambda_: Union[FloatTensorLike, Callable] = 0.01,
        epsilon: FloatTensorLike = 1e-7,
        ord: str = "fro",
        name: str = "ConditionalGradient",
        **kwargs,
    ):
        """Construct a new conditional gradient optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            lambda_: A `Tensor` or a floating point value. The constraint.
            epsilon: A `Tensor` or a floating point value. A small constant
                for numerical stability when handling the case of norm of
                gradient to be zero.
            ord: Order of the norm. Supported values are `'fro'`
                and `'nuclear'`. Default is `'fro'`, which is frobenius norm.
            name: Optional name prefix for the operations created when
                applying gradients. Defaults to 'ConditionalGradient'.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("lambda_", lambda_)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        supported_norms = ["fro", "nuclear"]
        if ord not in supported_norms:
            raise ValueError(
                "'ord' must be a supported matrix norm in %s, got '%s' instead"
                % (supported_norms, ord)
            )
        self.ord = ord

    def get_config(self):
        config = {
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "lambda_": self._serialize_hyperparameter("lambda_"),
            "epsilon": self.epsilon,
            "ord": self.ord,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_slots(self, var_list):
        for v in var_list:
            self.add_slot(v, "conditional_gradient")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state[(var_device, var_dtype)]["learning_rate"] = tf.identity(
            self._get_hyper("learning_rate", var_dtype)
        )
        apply_state[(var_device, var_dtype)]["lambda_"] = tf.identity(
            self._get_hyper("lambda_", var_dtype)
        )
        apply_state[(var_device, var_dtype)]["epsilon"] = tf.convert_to_tensor(
            self.epsilon, var_dtype
        )

    @staticmethod
    def _frobenius_norm(m):
        return tf.reduce_sum(m ** 2) ** 0.5

    @staticmethod
    def _top_singular_vector(m):
        # handle the case where m is a tensor of rank 0 or rank 1.
        # Example:
        #   scalar (rank 0) a, shape []=> [[a]], shape [1,1]
        #   vector (rank 1) [a,b], shape [2] => [[a,b]], shape [1,2]
        original_rank = tf.rank(m)
        shape = tf.shape(m)
        first_pad = tf.cast(tf.less(original_rank, 2), dtype=tf.int32)
        second_pad = tf.cast(tf.equal(original_rank, 0), dtype=tf.int32)
        new_shape = tf.concat(
            [
                tf.ones(shape=first_pad, dtype=tf.int32),
                tf.ones(shape=second_pad, dtype=tf.int32),
                shape,
            ],
            axis=0,
        )
        n = tf.reshape(m, new_shape)
        st, ut, vt = tf.linalg.svd(n, full_matrices=False)
        n_size = tf.shape(n)
        ut = tf.reshape(ut[:, 0], [n_size[0], 1])
        vt = tf.reshape(vt[:, 0], [n_size[1], 1])
        st = tf.matmul(ut, tf.transpose(vt))
        # when we return the top singular vector, we have to remove the
        # dimension we have added on
        st_shape = tf.shape(st)
        begin = tf.cast(tf.less(original_rank, 2), dtype=tf.int32)
        end = 2 - tf.cast(tf.equal(original_rank, 0), dtype=tf.int32)
        new_shape = st_shape[begin:end]
        return tf.reshape(st, new_shape)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        lr = coefficients["learning_rate"]
        lambda_ = coefficients["lambda_"]
        epsilon = coefficients["epsilon"]
        if self.ord == "fro":
            norm = tf.convert_to_tensor(
                self._frobenius_norm(grad), name="norm", dtype=var.dtype.base_dtype
            )
            s = grad / (norm + epsilon)
        else:
            top_singular_vector = tf.convert_to_tensor(
                self._top_singular_vector(grad),
                name="top_singular_vector",
                dtype=var.dtype.base_dtype,
            )
            s = top_singular_vector

        var_update = tf.math.multiply(var, lr) - (1 - lr) * lambda_ * s
        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        lr = coefficients["learning_rate"]
        lambda_ = coefficients["lambda_"]
        epsilon = coefficients["epsilon"]
        var_slice = tf.gather(var, indices)
        if self.ord == "fro":
            norm = tf.convert_to_tensor(
                self._frobenius_norm(grad), name="norm", dtype=var.dtype.base_dtype
            )
            s = grad / (norm + epsilon)
        else:
            top_singular_vector = tf.convert_to_tensor(
                self._top_singular_vector(grad),
                name="top_singular_vector",
                dtype=var.dtype.base_dtype,
            )
            s = top_singular_vector

        var_update_value = tf.math.multiply(var_slice, lr) - (1 - lr) * lambda_ * s
        var_update_op = self._resource_scatter_update(var, indices, var_update_value)
        return var_update_op
