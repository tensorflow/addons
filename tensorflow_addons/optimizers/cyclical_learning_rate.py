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
"""Cyclical Learning Rate Schedule policies for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked
from typing import Union, Callable


@tf.keras.utils.register_keras_serializable(package="Addons")
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses cyclical schedule."""

    @typechecked
    def __init__(
        self,
        initial_learning_rate: Union[FloatTensorLike, Callable],
        maximal_learning_rate: Union[FloatTensorLike, Callable],
        step_size: FloatTensorLike,
        scale_fn: Callable,
        scale_mode: str = "cycle",
        name: str = "CyclicalLearningRate",
    ):
        """Applies cyclical schedule to the learning rate.

        See Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186


        ```python
        lr_schedule = tf.keras.optimizers.schedules.CyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_fn=lambda x: 1.,
            scale_mode="cycle",
            name="MyCyclicScheduler")

        model.compile(optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lr_schedule),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        You can pass this schedule directly into a
        `tf.keras.optimizers.Optimizer` as the learning rate.

        Args:
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The initial learning rate.
            maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The maximum learning rate.
            step_size: A scalar `float32` or `float64` `Tensor` or a
                Python number. Step size denotes the number of training iterations it takes to get to maximal_learning_rate.
            scale_fn: A function. Scheduling function applied in cycle
            scale_mode: ['cycle', 'iterations']. Mode to apply during cyclic
                schedule
            name: (Optional) Name for the operation.

        Returns:
            Updated learning rate value.
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.step_size = step_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

            mode_step = cycle if self.scale_mode == "cycle" else step

            return initial_learning_rate + (
                maximal_learning_rate - initial_learning_rate
            ) * tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(mode_step)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "scale_fn": self.scale_fn,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }


@tf.keras.utils.register_keras_serializable(package="Addons")
class TriangularCyclicalLearningRate(CyclicalLearningRate):
    @typechecked
    def __init__(
        self,
        initial_learning_rate: Union[FloatTensorLike, Callable],
        maximal_learning_rate: Union[FloatTensorLike, Callable],
        step_size: FloatTensorLike,
        scale_mode: str = "cycle",
        name: str = "TriangularCyclicalLearningRate",
    ):
        """Applies triangular cyclical schedule to the learning rate.

        See Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186


        ```python
        from tf.keras.optimizers import schedules

        lr_schedule = schedules.TriangularCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_mode="cycle",
            name="MyCyclicScheduler")

        model.compile(optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lr_schedule),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        You can pass this schedule directly into a
        `tf.keras.optimizers.Optimizer` as the learning rate.

        Args:
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The initial learning rate.
            maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The maximum learning rate.
            step_size: A scalar `float32` or `float64` `Tensor` or a
                Python number. Step size denotes the number of training iterations it takes to get to maximal_learning_rate
            scale_mode: ['cycle', 'iterations']. Mode to apply during cyclic
                schedule
            name: (Optional) Name for the operation.

        Returns:
            Updated learning rate value.
        """
        super().__init__(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
            scale_fn=lambda x: 1.0,
            scale_mode=scale_mode,
            name=name,
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }


@tf.keras.utils.register_keras_serializable(package="Addons")
class Triangular2CyclicalLearningRate(CyclicalLearningRate):
    @typechecked
    def __init__(
        self,
        initial_learning_rate: Union[FloatTensorLike, Callable],
        maximal_learning_rate: Union[FloatTensorLike, Callable],
        step_size: FloatTensorLike,
        scale_mode: str = "cycle",
        name: str = "Triangular2CyclicalLearningRate",
    ):
        """Applies triangular2 cyclical schedule to the learning rate.

        See Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186


        ```python
        from tf.keras.optimizers import schedules

        lr_schedule = schedules.Triangular2CyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_mode="cycle",
            name="MyCyclicScheduler")

        model.compile(optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lr_schedule),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        You can pass this schedule directly into a
        `tf.keras.optimizers.Optimizer` as the learning rate.

        Args:
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The initial learning rate.
            maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The maximum learning rate.
            step_size: A scalar `float32` or `float64` `Tensor` or a
                Python number. Step size denotes the number of training iterations it takes to get to maximal_learning_rate
            scale_mode: ['cycle', 'iterations']. Mode to apply during cyclic
                schedule
            name: (Optional) Name for the operation.

        Returns:
            Updated learning rate value.
        """
        super().__init__(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
            scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
            scale_mode=scale_mode,
            name=name,
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }


@tf.keras.utils.register_keras_serializable(package="Addons")
class ExponentialCyclicalLearningRate(CyclicalLearningRate):
    @typechecked
    def __init__(
        self,
        initial_learning_rate: Union[FloatTensorLike, Callable],
        maximal_learning_rate: Union[FloatTensorLike, Callable],
        step_size: FloatTensorLike,
        scale_mode: str = "iterations",
        gamma: FloatTensorLike = 1.0,
        name: str = "ExponentialCyclicalLearningRate",
    ):
        """Applies exponential cyclical schedule to the learning rate.

        See Cyclical Learning Rates for Training Neural Networks. https://arxiv.org/abs/1506.01186


        ```python
        from tf.keras.optimizers import schedules

        lr_schedule = ExponentialCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=2000,
            scale_mode="cycle",
            gamma=0.96,
            name="MyCyclicScheduler")

        model.compile(optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lr_schedule),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        You can pass this schedule directly into a
        `tf.keras.optimizers.Optimizer` as the learning rate.

        Args:
            initial_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The initial learning rate.
            maximal_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The maximum learning rate.
            step_size: A scalar `float32` or `float64` `Tensor` or a
                Python number. Step size denotes the number of training iterations it takes to get to maximal_learning_rate
            scale_mode: ['cycle', 'iterations']. Mode to apply during cyclic
                schedule
            gamma: A scalar `float32` or `float64` `Tensor` or a
                Python number.  Gamma value.
            name: (Optional) Name for the operation.

        Returns:
            Updated learning rate value.
        """
        self.gamma = gamma
        super().__init__(
            initial_learning_rate=initial_learning_rate,
            maximal_learning_rate=maximal_learning_rate,
            step_size=step_size,
            scale_fn=lambda x: gamma ** x,
            scale_mode=scale_mode,
            name=name,
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
            "gamma": self.gamma,
        }
