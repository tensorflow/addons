# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import tensorflow as tf
from typeguard import typechecked
from tensorflow_addons.optimizers.average_wrapper import AveragedOptimizerWrapper


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    r"""The callback that saves average model weights.

    The callback that should be used with optimizers that extend
    `tfa.optimizers.AveragedOptimizerWrapper`, i.e.,
    `tfa.optimizers.MovingAverage` and
    `tfa.optimizers.StochasticAverage` optimizers.
    It saves and, optionally, assigns the averaged weights.

    Args:
        update_weights: If `True`, assign the moving average weights
            to the model, and save them. If False, keep the old
            non-averaged weights, but the saved model uses the
            average weights.

        See `tf.keras.callbacks.ModelCheckpoint` for the other args.
    """

    @typechecked
    def __init__(
        self,
        update_weights: bool,
        filepath: str,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = "auto",
        save_freq: str = "epoch",
        **kwargs
    ):
        self.update_weights = update_weights
        super().__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            **kwargs,
        )

    def set_model(self, model):
        if not isinstance(model.optimizer, AveragedOptimizerWrapper):
            raise TypeError(
                "AverageModelCheckpoint is only used when training"
                "with MovingAverage or StochasticAverage"
            )
        return super().set_model(model)

    def _save_model(self, epoch, logs):
        assert isinstance(self.model.optimizer, AveragedOptimizerWrapper)

        if self.update_weights:
            self.model.optimizer.assign_average_vars(self.model.variables)
            return super()._save_model(epoch, logs)
        else:
            # Note: `model.get_weights()` gives us the weights (non-ref)
            # whereas `model.variables` returns references to the variables.
            non_avg_weights = self.model.get_weights()
            self.model.optimizer.assign_average_vars(self.model.variables)
            # result is currently None, since `super._save_model` doesn't
            # return anything, but this may change in the future.
            result = super()._save_model(epoch, logs)
            self.model.set_weights(non_avg_weights)
            return result
