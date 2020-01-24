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
from tensorflow_addons.optimizers.average_wrapper import AveragedOptimizerWrapper


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 optimizer,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 **kwargs):
        if not isinstance(optimizer, AveragedOptimizerWrapper):
            raise TypeError(
                "AverageModelCheckpoint is only used when training with MovingAverage or StochasticAverage optimizers")

        self.optimizer = optimizer
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, **kwargs)

    def _save_model(self, epoch, logs):
        self.optimizer.assign_average_vars(self.model.variables)
        return super()._save_model(epoch, logs)
