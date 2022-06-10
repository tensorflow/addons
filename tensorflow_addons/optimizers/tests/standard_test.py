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
# ==============================================================================

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons import optimizers
from tensorflow_addons.optimizers import KerasLegacyOptimizer
from tensorflow_addons.utils.test_utils import discover_classes

class_exceptions = [
    "MultiOptimizer",  # is wrapper
    "SGDW",  # is wrapper
    "AdamW",  # is wrapper
    "SWA",  # is wrapper
    "AveragedOptimizerWrapper",  # is wrapper
    "ConditionalGradient",  # is wrapper
    "Lookahead",  # is wrapper
    "MovingAverage",  # is wrapper
    "KerasLegacyOptimizer",  # is a constantc
]

classes_to_test = discover_classes(optimizers, KerasLegacyOptimizer, class_exceptions)


@pytest.mark.parametrize("optimizer", classes_to_test)
@pytest.mark.parametrize("serialize", [True, False])
def test_optimizer_minimize_serialize(optimizer, serialize, tmpdir):
    """
    Purpose of this test is to confirm that the optimizer can minimize the loss in toy conditions.
    It also tests for serialization as a parameter.
    """
    model = tf.keras.Sequential([tf.keras.Input(shape=[1]), tf.keras.layers.Dense(1)])

    x = np.array(np.ones([1]))
    y = np.array(np.zeros([1]))

    opt = optimizer()
    loss = tf.keras.losses.MSE

    model.compile(optimizer=opt, loss=loss)

    # serialize whole model including optimizer, clear the session, then reload the whole model.
    # successfully serialized optimizers should not require a compile before training
    if serialize:
        model.save(str(tmpdir), save_format="tf")
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(str(tmpdir))

    history = model.fit(x, y, batch_size=1, epochs=10)

    loss_values = history.history["loss"]

    np.testing.assert_array_less(loss_values[-1], loss_values[0])
