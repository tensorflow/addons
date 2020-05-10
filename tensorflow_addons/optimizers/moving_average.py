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

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow_addons.optimizers import AveragedOptimizerWrapper
from tensorflow_addons.utils import types

from typing import Optional
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class MovingAverage(AveragedOptimizerWrapper):
    """Optimizer that computes a moving average of the variables.

    Empirically it has been found that using the moving average of the trained
    parameters of a deep network is better than using its trained parameters
    directly. This optimizer allows you to compute this moving average and swap
    the variables at save time so that any code outside of the training loop
    will use by default the average values instead of the original ones.

    Example of usage:

    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.MovingAverage(opt)

    ```
    """

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        sequential_update: bool = True,
        average_decay: types.FloatTensorLike = 0.99,
        num_updates: Optional[str] = None,
        name: str = "MovingAverage",
        **kwargs
    ):
        r"""Construct a new MovingAverage optimizer.

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            sequential_update: Bool. If False, will compute the moving average
                at the same time as the model is updated, potentially doing
                benign data races. If True, will update the moving average
                after gradient updates.
            average_decay: float. Decay to use to maintain the moving averages
                of trained variables.
            num_updates: Optional count of the number of updates applied to
                variables.
            name: Optional name for the operations created when applying
                gradients. Defaults to "MovingAverage".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(optimizer, sequential_update, name, **kwargs)
        self._num_updates = num_updates
        if self._num_updates is not None:
            num_updates = tf.cast(self._num_updates, tf.float32, name="num_updates")
            average_decay = tf.minimum(
                average_decay, (1.0 + num_updates) / (10.0 + num_updates)
            )

        self._set_hyper("average_decay", average_decay)

    def average_op(self, var, average_var):
        decay = self._get_hyper("average_decay", tf.dtypes.float32)
        return assign_moving_average(average_var, var, decay, False)

    def get_config(self):
        config = {
            "average_decay": self._serialize_hyperparameter("average_decay"),
            "num_updates": self._num_updates,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_slots(self, var_list):
        self._optimizer._create_slots(
            var_list=var_list
        )  # pylint: disable=protected-access
        for var in var_list:
            self.add_slot(var, "average", var.read_value())
