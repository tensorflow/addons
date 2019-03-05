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
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils as keras_test_util
from tensorflow_addons.layers.python import wrappers


class WeightNormalizationTest(tf.test.TestCase):
    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_dense_train(self):
        model = tf.keras.models.Sequential()
        model.add(
            wrappers.WeightNormalization(
                tf.keras.layers.Dense(2), input_shape=(3, 4)))

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='mse')
        model.fit(
            np.random.random((10, 3, 4)),
            np.random.random((10, 3, 2)),
            epochs=3,
            batch_size=10)
        self.assertTrue(hasattr(model.layers[0].layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_dense_train_notinit(self):
        model = tf.keras.models.Sequential()
        model.add(
            wrappers.WeightNormalization(
                tf.keras.layers.Dense(2), input_shape=(3, 4), data_init=False))

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
            loss='mse')
        model.fit(
            np.random.random((10, 3, 4)),
            np.random.random((10, 3, 2)),
            epochs=3,
            batch_size=10)
        self.assertTrue(hasattr(model.layers[0].layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_conv2d(self):
        model = tf.keras.models.Sequential()
        model.add(
            wrappers.WeightNormalization(
                tf.keras.layers.Conv2D(5, (2, 2), padding='same'),
                input_shape=(4, 4, 3)))

        model.add(tf.keras.layers.Activation('relu'))
        model.compile(
            optimizer=tf.optimizers.RMSprop(learning_rate=0.001), loss='mse')
        model.fit(
            np.random.random((2, 4, 4, 3)),
            np.random.random((2, 4, 4, 5)),
            epochs=3,
            batch_size=10)

        self.assertTrue(hasattr(model.layers[0].layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_tflayers(self):
        images = tf.random.uniform((2, 4, 4, 3))
        wn_wrapper = wrappers.WeightNormalization(
            tf.keras.layers.Conv2D(32, [2, 2]), input_shape=(4, 4, 3))
        wn_wrapper.apply(images)
        self.assertTrue(hasattr(wn_wrapper.layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_nonlayer(self):
        images = tf.random.uniform((2, 4, 43))
        with self.assertRaises(ValueError):
            wrappers.WeightNormalization(images)

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_nokernel(self):
        with self.assertRaises(ValueError):
            wrappers.WeightNormalization(tf.keras.layers.MaxPooling2D(
                2, 2)).build((2, 2))

    def test_weightnorm_keras(self):
        input_data = np.random.random((10, 3, 4)).astype(np.float32)
        outputs = keras_test_util.layer_test(
            wrappers.WeightNormalization,
            kwargs={
                'layer': tf.keras.layers.Dense(2),
                'input_shape': (3, 4)
            },
            input_data=input_data)


if __name__ == "__main__":
    tf.test.main()
