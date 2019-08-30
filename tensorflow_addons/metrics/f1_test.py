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
"""Tests F1 metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow_addons.utils import test_utils
from tensorflow.keras import layers
import numpy as np


@test_utils.run_all_in_graph_and_eager_modes
class F1ScoreTest(tf.test.TestCase):
    def test_config(self):
        f1_obj = F1Score(num_classes=3, average=None)
        self.assertEqual(f1_obj.name, 'f1_score')
        self.assertEqual(f1_obj.dtype, tf.float32)
        self.assertEqual(f1_obj.num_classes, 3)
        self.assertEqual(f1_obj.average, None)
        # Check save and restore config
        f1_obj2 = F1Score.from_config(f1_obj.get_config())
        self.assertEqual(f1_obj2.name, 'f1_score')
        self.assertEqual(f1_obj2.dtype, tf.float32)
        self.assertEqual(f1_obj2.num_classes, 3)
        self.assertEqual(f1_obj2.average, None)

    def initialize_vars(self, average):
        # initialize variables
        f1_obj = F1Score(num_classes=3, average=average)

        self.evaluate(tf.compat.v1.variables_initializer(f1_obj.variables))

        return f1_obj

    def update_obj_states(self, f1_obj, actuals, preds):
        # update state variable values
        update_op = f1_obj.update_state(actuals, preds)
        self.evaluate(update_op)

    def check_results(self, f1_obj, value):
        # check result
        self.assertAllClose(value, self.evaluate(f1_obj.result()), atol=1e-5)

    def _test_f1(self, avg, act, pred, res):
        f1_init = self.initialize_vars(avg)
        self.update_obj_states(f1_init, act, pred)
        self.check_results(f1_init, res)

    def _test_f1_score(self, actuals, preds, res):
        # test for three average values with beta value as 1.0
        for avg in ['micro', 'macro', 'weighted']:
            self._test_f1(avg, actuals, preds, res)

    # test for perfect f1 score
    def test_f1_perfect_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=tf.int32)
        self._test_f1_score(actuals, preds, 1.0)

    # test for worst f1 score
    def test_f1_worst_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int32)
        self._test_f1_score(actuals, preds, 0.0)

    # test for random f1 score
    def test_f1_random_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.int32)
        # Use absl parameterized test here if possible
        test_params = [['micro', 0.6666667], ['macro', 0.65555555],
                       ['weighted', 0.67777777]]

        for avg, res in test_params:
            self._test_f1(avg, actuals, preds, res)

    # test for random f1 score with average as None
    def test_f1_random_score_none(self):
        actuals = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=tf.int32)
        preds = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
            dtype=tf.int32)

        # Use absl parameterized test here if possible
        test_params = [[None, [0.8, 0.6666667, 1.]]]

        for avg, res in test_params:
            self._test_f1(avg, actuals, preds, res)

    # Keras model check
    def test_keras_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        fb = F1Score(1, 'macro')
        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['acc', fb])
        # data preparation
        data = np.random.random((10, 3))
        labels = np.random.random((10, 1))
        labels = np.where(labels > 0.5, 1, 0)
        model.fit(data, labels, epochs=1, batch_size=32, verbose=0)


if __name__ == '__main__':
    tf.test.main()
