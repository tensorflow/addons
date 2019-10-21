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
"""Tests for One Cycle Learnin rate policy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import test_utils
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import OneCycleLR


class OneCycleLRTest(tf.test.TestCase):
    # @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def doTestBasic(self):
        with self.cached_session():
            np.random.seed(1729)
            x_train = tf.random.uniform(shape=(160, 1))
            y_train = 2*x_train + 5
            model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
            model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

            cbks = [tf.keras.callbacks.OneCycleLR()]
            model.fit(x_train,
                      y_train,
                      batch_size=16,
                      callbacks=cbks,
                      epochs=10,
                      verbose=0)

            self.assertAllClose(float(
                tf.keras.backend.get_value(cbks[0].clr_iterations)),
                                1e-5,
                                atol=tf.keras.backend.epsilon())

            self.assertAllClose(float(
                tf.keras.backend.get_value(model.optimizer.lr)),
                                1e-5,
                                atol=tf.keras.backend.epsilon())
