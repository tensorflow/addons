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

from tensorflow_addons.utils import keras_utils
from tensorflow_addons.utils import test_utils


def _train_some_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

    x = np.ones(shape = [32, 1])
    y = np.ones(shape = [32, 1])
    model.compile(loss = 'binary_crossentropy')
    model.fit(x, y, epochs = 5)

    return model


@test_utils.run_all_distributed(2)
class NormalizeTupleTest(tf.test.TestCase):
    def test_train_model(self):
        _train_some_model()
        self.assertAllClose(1, 1)

    def test_normalize_tuple(self):
        self.assertEqual((2, 2, 2), keras_utils.normalize_tuple(2, n=3, name="strides"))
        self.assertEqual(
            (2, 1, 2), keras_utils.normalize_tuple((2, 1, 2), n=3, name="strides")
        )

        with self.assertRaises(ValueError):
            keras_utils.normalize_tuple((2, 1), n=3, name="strides")

        with self.assertRaises(TypeError):
            keras_utils.normalize_tuple(None, n=3, name="strides")


class AssertRNNCellTest(tf.test.TestCase):

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def test_model(self):
        _train_some_model()
        _train_some_model()
        self.assertAllClose(1, 1)

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def test_model_fail(self):
        _train_some_model()
        _train_some_model()
        self.assertAllClose(1, 0)

    @test_utils.run_distributed(2)
    def test_standard_cell(self):
        keras_utils.assert_like_rnncell("cell", tf.keras.layers.LSTMCell(10))

    @test_utils.run_distributed(2)
    def test_non_cell(self):
        with self.assertRaises(TypeError):
            keras_utils.assert_like_rnncell("cell", tf.keras.layers.Dense(10))

    def test_custom_cell(self):
        class CustomCell(tf.keras.layers.AbstractRNNCell):
            @property
            def output_size(self):
                raise ValueError("assert_like_rnncell should not run code")

        keras_utils.assert_like_rnncell("cell", CustomCell())


if __name__ == "__main__":
    _train_some_model()

    tf.test.main()
