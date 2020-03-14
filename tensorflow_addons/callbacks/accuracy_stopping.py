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
"""Callback that stops training when a specified amount of accuracy has achieved."""


import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typeguard import typechecked

@tf.keras.utils.register_keras_serializable(package="Addons")
class AccuracyStopping(Callback):
    """Stop training when a specified accuracy is reached.
    Args:
        Acc: Maximum accuracy before stopping.
            Defaults to 0.9999 %. It takes value in between 0 - 1. 
    """

    @typechecked
    def __init__(self, Acc: float = 0.9999):
        super().__init__()
        self.Acc = Acc
   

    def on_epoch_end(self, epoch, logs={}):
        print(self.Acc)
        print(logs.get('accuracy'))
        if (logs.get('accuracy') >= self.Acc):
            self.model.stop_training = True
            self.stopped_epoch = epoch
            

    def on_train_end(self, logs=None):
        if self.stopped_epoch is not None:
            msg = "Reached {} % accuracy on {} epochs so cancelling training!" .format(
                self.Acc*100, self.stopped_epoch + 1)
            print(msg)
            

    def get_config(self):
        config = {
            "Acc": self.Acc,
            "verbose": self.verbose,
        }

        base_config = super().get_config()
        return {**base_config, **config}
    
