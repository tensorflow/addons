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
from tensorflow_addons.layers.python import wrappers

from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.layers import layers
from tensorflow.python.training.rmsprop import RMSPropOptimizer

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python import keras


class WeightNormTest(test.TestCase):

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_dense_train(self):
        model = keras.models.Sequential()
        model.add(wrappers.WeightNorm(
            keras.layers.Dense(2), input_shape=(3, 4)))

        model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
        model.fit(
            np.random.random((10, 3, 4)),
            np.random.random((10, 3, 2)),
            epochs=1,
            batch_size=10)
        self.assertTrue(hasattr(model.layers[0].layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weightnorm_conv2d(self):
        model = keras.models.Sequential()
        model.add(wrappers.WeightNorm(
            keras.layers.Conv2D(5, (2, 2), padding='same'),
            input_shape=(4, 4, 3)))

        model.add(keras.layers.Activation('relu'))
        model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
        model.train_on_batch(
            np.random.random((2, 4, 4, 3)),
            np.random.random((2, 4, 4, 5)))

        self.assertTrue(hasattr(model.layers[0].layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weight_norm_tflayers(self):
        images = random_ops.random_uniform((2, 4, 4, 3))
        wn_wrapper = wrappers.WeightNorm(layers.Conv2D(32, [2, 2]),
                                         input_shape=(4, 4, 3))
        wn_wrapper.apply(images)
        self.assertTrue(hasattr(wn_wrapper.layer, 'g'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weight_norm_nonlayer(self):
        images = random_ops.random_uniform((2, 4, 43))
        with self.assertRaises(ValueError):
            wrappers.WeightNorm(images)

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weight_norm_nokernel(self):
        with self.assertRaises(ValueError):
            wrappers.WeightNorm(layers.MaxPooling2D(2, 2)).build((2, 2))


if __name__ == "__main__":
    test.main()
