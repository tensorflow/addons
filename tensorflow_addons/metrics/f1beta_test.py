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
"""Tests F1 beta metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.metrics import FBetaScore
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class FBetaScoreTest(tf.test.TestCase):
    def test_config(self):
        fbeta_obj = FBetaScore(num_classes=3, beta=0.5, average=None)
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.average, None)
        self.assertEqual(fbeta_obj.num_classes, 3)
        # Check save and restore config
        fbeta_obj2 = FBetaScore.from_config(fbeta_obj.get_config())
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.average, None)
        self.assertEqual(fbeta_obj2.num_classes, 3)

    def initialize_vars(self):
        fbeta_micro = FBetaScore(num_classes=3,
                                 beta=0.5, average='micro')
        fbeta_macro = FBetaScore(num_classes=3,
                                 beta=0.5, average='macro')
        fbeta_weighted = FBetaScore(num_classes=3,
                                    beta=0.5, average='weighted')

        self.evaluate(tf.compat.v1.variables_initializer(
                      fbeta_micro.variables))
        self.evaluate(tf.compat.v1.variables_initializer(
                      fbeta_macro.variables))
        self.evaluate(
            tf.compat.v1.variables_initializer(fbeta_weighted.variables))
        return fbeta_micro, fbeta_macro, fbeta_weighted

    def initialize_vars_none(self):
        fbeta_none = FBetaScore(num_classes=3, beta=0.5, average=None)

        self.evaluate(tf.compat.v1.variables_initializer(fbeta_none.variables))
        return fbeta_none

    def update_obj_states(self, fbeta_micro, fbeta_macro, fbeta_weighted,
                          actuals, preds):
        update_micro = fbeta_micro.update_state(actuals, preds)
        update_macro = fbeta_macro.update_state(actuals, preds)
        update_weighted = fbeta_weighted.update_state(actuals, preds)
        self.evaluate(update_micro)
        self.evaluate(update_macro)
        self.evaluate(update_weighted)

    def update_obj_states_none(self, fbeta_none, actuals, preds):
        update_none = fbeta_none.update_state(actuals, preds)
        self.evaluate(update_none)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_fbeta_perfect_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=tf.int32)
        # Initialize
        fbeta_micro, fbeta_macro, fbeta_weighted = self.initialize_vars()
        # Update
        self.update_obj_states(fbeta_micro, fbeta_macro, fbeta_weighted,
                               actuals, preds)
        # Check results
        self.check_results(fbeta_micro, 1.0)
        self.check_results(fbeta_macro, 1.0)
        self.check_results(fbeta_weighted, 1.0)

    def test_fbeta_worst_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int32)
        # Initialize
        fbeta_micro, fbeta_macro, fbeta_weighted = self.initialize_vars()
        # Update
        self.update_obj_states(fbeta_micro, fbeta_macro, fbeta_weighted,
                               actuals, preds)
        # Check results
        self.check_results(fbeta_micro, 0.0)
        self.check_results(fbeta_macro, 0.0)
        self.check_results(fbeta_weighted, 0.0)

    def test_fbeta_random_score(self):
        actuals = tf.constant([[1, 1, 1], [1, 0, 0], [1, 1, 0]],
                              dtype=tf.int32)
        preds = tf.constant([[0, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tf.int32)
        # Initialize
        fbeta_micro, fbeta_macro, fbeta_weighted = self.initialize_vars()
        # Update
        self.update_obj_states(fbeta_micro, fbeta_macro, fbeta_weighted,
                               actuals, preds)
        # Check results
        self.check_results(fbeta_micro, 0.6666667)
        self.check_results(fbeta_macro, 0.6548822)
        self.check_results(fbeta_weighted, 0.7138047)

    def test_fbeta_none_score(self):
        actuals = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=tf.int32)
        preds = tf.constant(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
            dtype=tf.int32)
        # Initialize
        fbeta_none = self.initialize_vars_none()
        # Update
        self.update_obj_states_none(fbeta_none, actuals, preds)
        # Check results
        self.check_results(fbeta_none, [0.71428573, 0.8333334, 1.])


if __name__ == '__main__':
    tf.test.main()
