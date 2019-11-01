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

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_addons.metrics import FBetaScore, F1Score, utils
from tensorflow_addons.utils import test_utils

import numpy as np
from sklearn.metrics import fbeta_score


@test_utils.run_all_in_graph_and_eager_modes
class FBetaScoreTest(tf.test.TestCase, parameterized.TestCase):
    def test_config(self):
        fbeta_obj = FBetaScore(
            num_classes=3, beta=0.5, threshold=0.3, average=None)
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.average, None)
        self.assertEqual(fbeta_obj.threshold, 0.3)
        self.assertEqual(fbeta_obj.num_classes, 3)
        self.assertEqual(fbeta_obj.dtype, tf.float32)

        # Check save and restore config
        fbeta_obj2 = FBetaScore.from_config(fbeta_obj.get_config())
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.average, None)
        self.assertEqual(fbeta_obj2.threshold, 0.3)
        self.assertEqual(fbeta_obj2.num_classes, 3)
        self.assertEqual(fbeta_obj2.dtype, tf.float32)

    def _test_tf(self, avg, beta, act, pred, threshold):
        act = tf.constant(act, tf.float32)
        pred = tf.constant(pred, tf.float32)

        fbeta = FBetaScore(3, avg, beta, threshold)
        self.evaluate(tf.compat.v1.variables_initializer(fbeta.variables))
        self.evaluate(fbeta.update_state(act, pred))
        return self.evaluate(fbeta.result())

    def _test_fbeta_score(self, actuals, preds, avg, beta_val, result,
                          threshold):
        tf_score = self._test_tf(avg, beta_val, actuals, preds, threshold)
        self.assertAllClose(tf_score, result, atol=1e-7)

    def test_fbeta_perfect_score(self):
        preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        actuals = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]

        for avg_val in ['micro', 'macro', 'weighted']:
            for beta in [0.5, 1.0, 2.0]:
                self._test_fbeta_score(actuals, preds, avg_val, beta, 1.0,
                                       0.66)

    def test_fbeta_worst_score(self):
        preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        actuals = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

        for avg_val in ['micro', 'macro', 'weighted']:
            for beta in [0.5, 1.0, 2.0]:
                self._test_fbeta_score(actuals, preds, avg_val, beta, 0.0,
                                       0.66)

    @parameterized.parameters([[None, 0.5, [0.71428573, 0.5, 0.833334]],
                               [None, 1.0, [0.8, 0.5, 0.6666667]],
                               [None, 2.0, [0.9090904, 0.5, 0.555556]],
                               ['micro', 0.5, 0.6666667],
                               ['micro', 1.0, 0.6666667],
                               ['micro', 2.0, 0.6666667],
                               ['macro', 0.5, 0.6825397],
                               ['macro', 1.0, 0.6555555],
                               ['macro', 2.0, 0.6548822],
                               ['weighted', 0.5, 0.6825397],
                               ['weighted', 1.0, 0.6555555],
                               ['weighted', 2.0, 0.6548822]])
    def test_fbeta_random_score(self, avg_val, beta, result):
        preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        actuals = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
        self._test_fbeta_score(actuals, preds, avg_val, beta, result, 0.66)

    @parameterized.parameters([[None, 0.5, [0.9090904, 0.555556, 1.0]],
                               [None, 1.0, [0.8, 0.6666667, 1.0]],
                               [None, 2.0, [0.71428573, 0.833334, 1.0]],
                               ['micro', 0.5, 0.833334],
                               ['micro', 1.0, 0.833334],
                               ['micro', 2.0, 0.833334],
                               ['macro', 0.5, 0.821549],
                               ['macro', 1.0, 0.822222],
                               ['macro', 2.0, 0.849206],
                               ['weighted', 0.5, 0.880471],
                               ['weighted', 1.0, 0.844445],
                               ['weighted', 2.0, 0.829365]])
    def test_fbeta_random_score_none(self, avg_val, beta, result):
        preds = [[0.9, 0.1, 0], [0.2, 0.6, 0.2], [0, 0, 1], [0.4, 0.3, 0.3],
                 [0, 0.9, 0.1], [0, 0, 1]]
        actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0],
                   [0, 0, 1]]
        self._test_fbeta_score(actuals, preds, avg_val, beta, result, None)

    def test_keras_model(self):
        fbeta = FBetaScore(5, 'micro', 1.0)
        utils.test_keras_model(fbeta, 5)


@test_utils.run_all_in_graph_and_eager_modes
class F1ScoreTest(tf.test.TestCase):
    def test_eq(self):
        f1 = F1Score(3)
        fbeta = FBetaScore(3, beta=1.0)
        self.evaluate(tf.compat.v1.variables_initializer(f1.variables))
        self.evaluate(tf.compat.v1.variables_initializer(fbeta.variables))

        preds = [[0.9, 0.1, 0], [0.2, 0.6, 0.2], [0, 0, 1], [0.4, 0.3, 0.3],
                 [0, 0.9, 0.1], [0, 0, 1]]
        actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0],
                   [0, 0, 1]]

        self.evaluate(fbeta.update_state(actuals, preds))
        self.evaluate(f1.update_state(actuals, preds))
        self.assertAllClose(
            self.evaluate(fbeta.result()), self.evaluate(f1.result()))

    def test_keras_model(self):
        f1 = F1Score(5)
        utils.test_keras_model(f1, 5)

    def test_config(self):
        f1 = F1Score(3)
        config = f1.get_config()
        self.assertFalse("beta" in config)


if __name__ == '__main__':
    tf.test.main()
