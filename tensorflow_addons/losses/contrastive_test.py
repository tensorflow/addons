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
"""Tests for contrastive loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.losses import contrastive
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class ContrastiveLossTest(tf.test.TestCase):

    def test_config(self):
        cl_obj = contrastive.ContrastiveLoss(
            reduction=tf.keras.losses.Reduction.SUM, name='cl')
        self.assertEqual(cl_obj.name, 'cl')
        self.assertEqual(cl_obj.reduction, tf.keras.losses.Reduction.SUM)

    def test_all_correct_unweighted(self):
        cl_obj = contrastive.ContrastiveLoss()
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([1., 1., 0., 0., 1., 0.],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cl_obj = contrastive.ContrastiveLoss()
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 0.6216, 3)

    def test_scalar_weighted(self):
        cl_obj = contrastive.ContrastiveLoss()
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred, sample_weight=6.0)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 3.73, 3)

    def test_sample_weighted(self):
        cl_obj = contrastive.ContrastiveLoss()
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        sample_weight = tf.constant([1.2, 0.8, 0.5, 0.4, 1.5, 1.0],
                                    dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred, sample_weight=sample_weight)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 0.4424, 3)

    def test_zero_weighted(self):
        cl_obj = contrastive.ContrastiveLoss()
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred, sample_weight=0.0)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_non_default_margin(self):
        cl_obj = contrastive.ContrastiveLoss(margin=2.0)
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 1.6233, 3)

    def test_no_reduction(self):
        cl_obj = contrastive.ContrastiveLoss(
            reduction=tf.keras.losses.Reduction.NONE)
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred)
        loss = self.evaluate(loss)
        self.assertArrayNear(loss, [0.81, 0.49, 1.69, 0.49, 0.0, 0.25], 1e-3)

    def test_sum_reduction(self):
        cl_obj = contrastive.ContrastiveLoss(
            reduction=tf.keras.losses.Reduction.SUM)
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([0.1, 0.3, 1.3, 0.7, 1.1, 0.5],
                             dtype=tf.dtypes.float32)
        loss = cl_obj(y_true, y_pred)
        loss = self.evaluate(loss)
        self.assertAlmostEqual(loss, 3.73, 3)


if __name__ == "__main__":
    tf.test.main()
