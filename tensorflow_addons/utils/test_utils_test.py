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
"""Utilities for tf.test.TestCase."""

import tensorflow as tf
import numpy as np
from tensorflow_addons.utils import test_utils

@test_utils.run_all_distributed
class DiscriminativeLearningTest(tf.test.TestCase):
    def test_distributed_1(self):


        train, test = tf.keras.datasets.cifar100.load_data()
        train_tanksandtrains = np.isin(train[1], [85, 90]).flatten()
        train_x = train[0][train_tanksandtrains]
        train_y = train[1][train_tanksandtrains]
        # if is tank then 1 else 0
        train_y = (train_y == 85) * 1

        model = tf.keras.Sequential()

        model.add(tf.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                          input_shape=(32, 32, 3),
                                                          include_top=False,
                                                          pooling='avg'))
        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation('sigmoid'))

        model.fit(train_x, train_y, batch_size=64, epochs = 1)

    def test_distributed_2(self):

        train, test = tf.keras.datasets.cifar100.load_data()
        train_tanksandtrains = np.isin(train[1], [85, 90]).flatten()
        train_x = train[0][train_tanksandtrains]
        train_y = train[1][train_tanksandtrains]
        # if is tank then 1 else 0
        train_y = (train_y == 85) * 1

        model = tf.keras.Sequential()

        model.add(tf.keras.applications.resnet50.ResNet50(weights='imagenet',
                                                          input_shape=(32, 32, 3),
                                                          include_top=False,
                                                          pooling='avg'))
        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation('sigmoid'))

        model.fit(train_x, train_y, batch_size=64, epochs = 1)


if __name__ == "__main__":
    tf.test.main()
