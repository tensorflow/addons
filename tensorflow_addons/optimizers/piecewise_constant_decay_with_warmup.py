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
"""Warmup Learning Rate with Step Decay Schedule policy for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, List, Number

from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class PiecewiseConstantDecayWithLinearWarmup(
    tf.keras.optimizers.schedules.PiecewiseConstantDecay
):
    """A LearningRateSchedule that applies linear warmup along with
    step decay schedule"""

    @typechecked
    def __init__(
        self,
        warmup_learning_rate: FloatTensorLike,
        warmup_steps: Number,
        boundaries: List,
        values: List,
        name: str = "PiecewiseConstantDecayWithLinearWarmup",
    ):

        """Applies linear warmup schedule for `warmup_steps` steps, and
        then follows `PiecewiseConstantDecay` schedule there after.

        ```python
        lr_schedule = \
            tf.keras.optimizers.schedules.PiecewiseConstantDecayWithLinearWarmup(
            warmup_learning_rate=0.0067,
            warmup_steps=500,
            boundaries=[60000, 80000],
            values=[0.01, 0.001, 0.0001])

        model.compile(optimizer=tf.keras.optimizers.SGD(
                                                    learning_rate=lr_schedule),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(data, labels, epochs=5)
        ```

        You can pass this schedule directly into a
        `tf.keras.optimizers.Optimizer` as the learning rate.

        Args:
            warmup_learning_rate: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The initial warmup learning rate.
            warmup_steps: A scalar `float32` or `float64` `Tensor` or
                a Python number.  The numer of warmup steps, during which the
                learning rates is increased linearly.
            boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
                increasing entries, and with all elements having the same type as the
                optimizer step.
            values: A list of `Tensor`s or `float`s or `int`s that specifies the
                values for the intervals defined by `boundaries`. It should have one
                more element than `boundaries`, and all elements should have the same
                type.
            name: A string. Optional name of the operation. Defaults to
                'PiecewiseConstantDecayWithLinearWarmup'.


        Returns:
            Updated learning rate value.

        Raises:
            ValueError: If the number of elements in the lists do not match.
        """

        super(PiecewiseConstantDecayWithLinearWarmup, self).__init__(
            boundaries=boundaries, values=values, name=name
        )
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self._step_size = self.values[0] - self.warmup_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "PiecewiseConstantDecayWithLinearWarmup"):
            if step < self.warmup_steps:
                learning_rate = (
                    self.warmup_learning_rate
                    + tf.cast(step, dtype=tf.float32)
                    / tf.cast(self.warmup_steps, dtype=tf.float32)
                    * self._step_size
                )
            else:
                learning_rate = super(
                    PiecewiseConstantDecayWithLinearWarmup, self
                ).__call__(step)
        return learning_rate

    def get_config(self):
        config = {
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        base_config = super(PiecewiseConstantDecayWithLinearWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
