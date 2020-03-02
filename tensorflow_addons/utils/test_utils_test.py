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
"""Tests for Keras utils."""

import tensorflow as tf
import numpy as np
from tensorflow_addons.utils import test_utils


def _train_something():

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

    model.comple(loss="mse", optimizer="sgd")

    x = np.zeros(shape=(32, 1))
    y = np.zeros(shape=(32, 1))

    model.fit(x, y, batch_size=2, epochs=2)


@test_utils.run_all_distributed(2)
class TestUtilsTest(tf.test.TestCase):
    def test_training(self):
        _train_something()

    def test_training_again(self):
        _train_something()

    @test_utils.run_in_graph_and_eager_modes
    def test_training_graph_eager(self):
        _train_something()


if __name__ == "__main__":
    tf.test.main()
