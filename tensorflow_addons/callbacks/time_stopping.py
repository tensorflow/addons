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
"""Callback that stops training when a specified amount of time has passed."""

from __future__ import absolute_import, division, print_function

import datetime
import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


@tf.keras.utils.register_keras_serializable(package='Addons')
class TimeStopping(Callback):
    """Stop training when a specified amount of time has passed.

    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
    """

    def __init__(self, seconds=86400, verbose=0):
        super(TimeStopping, self).__init__()

        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds

    def on_epoch_end(self, epoch, logs={}):
        if time.time() >= self.stopping_time:
            self.model.stop_training = True
            self.stopped_epoch = epoch

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = 'Timed stopping at epoch {} after training for {}'.format(
                self.stopped_epoch + 1, formatted_time)
            print(msg)

    def get_config(self):
        config = {
            'seconds': self.seconds,
            'verbose': self.verbose,
        }

        base_config = super(TimeStopping, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
