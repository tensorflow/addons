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
    # run a simple training loop to confirm that run distributed works.

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

    model.compile(loss="mse", optimizer="sgd")

    x = np.zeros(shape=(32, 1))
    y = np.zeros(shape=(32, 1))

    model.fit(x, y, batch_size=2, epochs=2)


class TestA(tf.test.TestCase):
    # hopefully this test will run first so things init properly.
    @test_utils.run_distributed(4)
    def test_training_dist(self):
        _train_something()


class TestUtilsTestMixed(tf.test.TestCase):
    # we should be able to run some tests that are distributed and some that are not distributed.
    def test_training(self):
        _train_something()

    @test_utils.run_distributed(4)
    def test_training_dist_many(self):
        _train_something()

    @test_utils.run_distributed(2)
    def test_training_dist_few(self):
        _train_something()

    @test_utils.run_in_graph_and_eager_modes
    def test_training_graph_eager(self):
        _train_something()

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def test_training_graph_eager_dist(self):
        _train_something()

    def test_train_dist_too_many(self):
        with self.assertRaises(RuntimeError):
            # create a function that is wrapped. if we wrapped test_train_dist_too_many, the error is raised
            # outside of the scope of self.assertRaises.
            func = test_utils.run_distributed(10)(self.test_training())
            func(self)
            # this should raise a runtime error


@test_utils.run_all_distributed(3)
class TestUtilsTest(tf.test.TestCase):
    # test the class wrapper
    def test_training(self):
        _train_something()

    def test_training_again(self):
        _train_something()


if __name__ == "__main__":
    tf.test.main()
