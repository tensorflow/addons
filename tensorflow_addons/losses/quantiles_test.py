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
"""Tests for pinball loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa


@tfa.utils.test_utils.run_all_in_graph_and_eager_modes
class PinballLossTest(tf.test.TestCase):
    def test_config(self):
        pin_obj = tfa.losses.PinballLoss(reduction=tf.keras.losses.Reduction.SUM, name='pin_1')
        self.assertEqual(pin_obj.name, 'pin_1')
        self.assertEqual(pin_obj.reduction, tf.keras.losses.Reduction.SUM)

    def test_all_correct_unweighted(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = pin_obj(y_true, y_true)
        self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    def test_unweighted(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 2.75, 3)

    def test_unweighted_quantile_0pc(self):
        pin_obj = tfa.losses.PinballLoss(tau=0.)
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 4.8333, 3)

    def test_unweighted_quantile_10pc(self):
        pin_obj = tfa.losses.PinballLoss(tau=.1)
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 4.4166, 3)

    def test_unweighted_quantile_90pc(self):
        pin_obj = tfa.losses.PinballLoss(tau=.9)
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 1.0833, 3)

    def test_unweighted_quantile_100pc(self):
        pin_obj = tfa.losses.PinballLoss(tau=1.)
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 0.6666, 3)

    def test_scalar_weighted(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(self.evaluate(loss), 6.325, 3)

    def test_sample_weighted(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        sample_weight = tf.constant([1.2, 3.4], shape=(2, 1))
        loss = pin_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(self.evaluate(loss), 40.7 / 6, 3)

    def test_timestep_weighted(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3, 1),
                             dtype=tf.dtypes.float32)
        sample_weight = tf.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = pin_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(self.evaluate(loss), 41.5 / 6, 3)

    def test_zero_weighted(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(self.evaluate(loss), 0.0, 3)

    def test_invalid_sample_weight(self):
        pin_obj = tfa.losses.PinballLoss()
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = tf.constant([3, 6, 5, 0], shape=(2, 2))
        with self.assertRaisesRegexp(ValueError,
                                     'weights can not be broadcast to values'):
            pin_obj(y_true, y_pred, sample_weight=sample_weight)

    def test_no_reduction(self):
        pin_obj = tfa.losses.PinballLoss(reduction=tf.keras.losses.Reduction.NONE)
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred, sample_weight=2.3)
        loss = self.evaluate(loss)
        self.assertArrayNear(loss, [5.3666, 7.28333], 1e-3)

    def test_sum_reduction(self):
        pin_obj = tfa.losses.PinballLoss(reduction=tf.keras.losses.Reduction.SUM)
        y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = tf.constant([4, 8, 12, 8, 1, 3],
                             shape=(2, 3),
                             dtype=tf.dtypes.float32)
        loss = pin_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(self.evaluate(loss), 12.65, 3)
