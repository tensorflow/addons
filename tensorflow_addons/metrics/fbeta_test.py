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
"""Tests F beta metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import FBetaScore
from tensorflow_addons.utils import test_utils
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np


@test_utils.run_all_in_graph_and_eager_modes
class FBetaScoreTest(tf.test.TestCase):
    def test_config(self):
        fbeta_obj = FBetaScore(num_classes=3, beta=0.5, average=None)
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.average, None)
        self.assertEqual(fbeta_obj.num_classes, 3)
        self.assertEqual(fbeta_obj.dtype, tf.float32)
        # Check save and restore config
        fbeta_obj2 = FBetaScore.from_config(fbeta_obj.get_config())
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.average, None)
        self.assertEqual(fbeta_obj2.num_classes, 3)
        self.assertEqual(fbeta_obj2.dtype, tf.float32)

    def initialize_vars(self, beta_val, average):
        # initialize variables
        fbeta_obj = FBetaScore(num_classes=3, beta=beta_val, average=average)

        self.evaluate(tf.compat.v1.variables_initializer(fbeta_obj.variables))

        return fbeta_obj

    def update_obj_states(self, fbeta_obj, actuals, preds):
        # update state variables values
        update_op = fbeta_obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, fbeta_obj, value):
        # check results
        self.assertAllClose(
            value, self.evaluate(fbeta_obj.result()), atol=1e-5)

    def _test_fbeta(self, avg, beta, act, pred, res):
        fbeta = self.initialize_vars(beta, avg)
        self.update_obj_states(fbeta, act, pred)
        self.check_results(fbeta, res)

    def _test_fbeta_score(self, actuals, preds, res):
        # This function tests for three average values and
        # two beta values
        for avg in ['micro', 'macro', 'weighted']:
            for beta_val in [0.5, 2.0]:
                self._test_fbeta(avg, beta_val, actuals, preds, res)

    # test for the perfect score
    def test_fbeta_perfect_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=tf.int32)
        self._test_fbeta_score(actuals, preds, 1.0)

    # test for the worst score
    def test_fbeta_worst_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int32)
        self._test_fbeta_score(actuals, preds, 0.0)

    # test for the random score
    def test_fbeta_random_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.int32)

        # test parameters
        test_params = [['micro', 0.5, 0.666667], ['macro', 0.5, 0.654882],
                       ['weighted', 0.5, 0.71380], ['micro', 2.0, 0.666667],
                       ['macro', 2.0, 0.68253], ['weighted', 2.0, 0.66269]]

        for avg, beta, res in test_params:
            self._test_fbeta(avg, beta, actuals, preds, res)

    # Test for the random score with average value as None
    def test_fbeta_random_score_none(self):
        actuals = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=tf.int32)
        preds = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
            dtype=tf.int32)

        # test parameters
        test_params = [[0.5, [0.71428573, 0.8333334, 1.]],
                       [2.0, [0.90909094, 0.5555556, 1.]]]

        for beta, res in test_params:
            self._test_fbeta(None, beta, actuals, preds, res)

    # Keras model check
    def keras_check(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        fb = FBetaScore(1, 'macro')
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['acc', fb])
        # data preparation
        data = np.random.random((10, 3))
        labels = np.random.random((10, 1))
        labels = np.where(labels > 0.5, 1, 0)
        model.fit(data, labels, epochs=10, batch_size=32, verbose=0)


if __name__ == '__main__':
    tf.test.main()
