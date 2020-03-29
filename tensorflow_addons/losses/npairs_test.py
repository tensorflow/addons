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
"""Tests for npairs loss."""

import sys
import platform
import unittest

import pytest
import tensorflow as tf
from tensorflow_addons.losses import npairs
from tensorflow_addons.utils import test_utils

IS_WINDOWS = platform.system() == "Windows"


@unittest.skipIf(
    IS_WINDOWS,
    reason="Doesn't work on Windows, see https://github.com/tensorflow/addons/issues/838",
)
@test_utils.run_all_in_graph_and_eager_modes
class NpairsLossTest(tf.test.TestCase):
    def test_config(self):
        nl_obj = npairs.NpairsLoss(name="nl")
        self.assertEqual(nl_obj.name, "nl")
        self.assertEqual(nl_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_unweighted(self):
        nl_obj = npairs.NpairsLoss()
        # batch size = 4, hidden size = 2
        y_true = tf.constant([0, 1, 2, 3], dtype=tf.int64)
        # features of anchors
        f = tf.constant(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32
        )
        # features of positive samples
        fp = tf.constant(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32
        )
        # similarity matrix
        y_pred = tf.matmul(f, fp, transpose_a=False, transpose_b=True)
        loss = nl_obj(y_true, y_pred)

        # Loss = 1/4 * \sum_i log(1 + \sum_{j != i} exp(f_i*fp_j^T-f_i*f_i^T))
        # Compute loss for i = 0, 1, 2, 3 without multiplier 1/4
        # i = 0 => log(1 + sum([exp(-2), exp(-2), exp(-4)])) = 0.253846
        # i = 1 => log(1 + sum([exp(-2), exp(-4), exp(-2)])) = 0.253846
        # i = 2 => log(1 + sum([exp(-2), exp(-4), exp(-2)])) = 0.253846
        # i = 3 => log(1 + sum([exp(-4), exp(-2), exp(-2)])) = 0.253846
        # Loss = (0.253856 + 0.253856 + 0.253856 + 0.253856) / 4 = 0.253856

        self.assertAllClose(loss, 0.253856)


@unittest.skipIf(
    IS_WINDOWS,
    reason="Doesn't work on Windows, see https://github.com/tensorflow/addons/issues/838",
)
@test_utils.run_all_in_graph_and_eager_modes
class NpairsMultilabelLossTest(tf.test.TestCase):
    def config(self):
        nml_obj = npairs.NpairsMultilabelLoss(name="nml")
        self.assertEqual(nml_obj.name, "nml")
        self.assertEqual(nml_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_single_label(self):
        """Test single label, which is the same with `NpairsLoss`."""
        nml_obj = npairs.NpairsMultilabelLoss()
        # batch size = 4, hidden size = 2
        y_true = tf.constant(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=tf.int64
        )
        # features of anchors
        f = tf.constant(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32
        )
        # features of positive samples
        fp = tf.constant(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32
        )
        # similarity matrix
        y_pred = tf.matmul(f, fp, transpose_a=False, transpose_b=True)
        loss = nml_obj(y_true, y_pred)

        # Loss = 1/4 * \sum_i log(1 + \sum_{j != i} exp(f_i*fp_j^T-f_i*f_i^T))
        # Compute loss for i = 0, 1, 2, 3 without multiplier 1/4
        # i = 0 => log(1 + sum([exp(-2), exp(-2), exp(-4)])) = 0.253846
        # i = 1 => log(1 + sum([exp(-2), exp(-4), exp(-2)])) = 0.253846
        # i = 2 => log(1 + sum([exp(-2), exp(-4), exp(-2)])) = 0.253846
        # i = 3 => log(1 + sum([exp(-4), exp(-2), exp(-2)])) = 0.253846
        # Loss = (0.253856 + 0.253856 + 0.253856 + 0.253856) / 4 = 0.253856

        self.assertAllClose(loss, 0.253856)

        # Test sparse tensor
        y_true = tf.sparse.from_dense(y_true)
        loss = nml_obj(y_true, y_pred)
        self.assertAllClose(loss, 0.253856)

    def test_multilabel(self):
        nml_obj = npairs.NpairsMultilabelLoss()
        # batch size = 4, hidden size = 2
        y_true = tf.constant(
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=tf.int64
        )
        # features of anchors
        f = tf.constant(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32
        )
        # features of positive samples
        fp = tf.constant(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]], dtype=tf.float32
        )
        # similarity matrix
        y_pred = tf.matmul(f, fp, transpose_a=False, transpose_b=True)
        loss = nml_obj(y_true, y_pred)

        # Loss = \sum_i log(1 + \sum_{j != i} exp(f_i*fp_j^T-f_i*f_i^T))
        # Because of multilabel, the label matrix is normalized so that each
        # row sums to one. That's why the multiplier before log exists.
        # Compute loss for i = 0, 1, 2, 3 without multiplier 1/4
        # i = 0 => 2/3 * log(1 + sum([exp(-2), exp(-2), exp(-4)])) +
        #          1/3 * log(1 + sum([exp(2) , exp(0) , exp(-2)])) = 0.920522
        # i = 1 => 1/4 * log(1 + sum([exp(2) , exp(-2), exp(0) ])) +
        #          1/2 * log(1 + sum([exp(-2), exp(-4), exp(-2)])) +
        #          1/4 * log(1 + sum([exp(2) , exp(4) , exp(2) ])) = 1.753856
        # i = 2 => 1/4 * log(1 + sum([exp(2) , exp(4) , exp(2) ])) +
        #          1/2 * log(1 + sum([exp(-2), exp(-4), exp(-2)])) +
        #          1/4 * log(1 + sum([exp(0) , exp(-2), exp(2) ])) = 1.753856
        # i = 4 => 1/2 * log(1 + sum([exp(-2), exp(0) , exp(2) ])) +
        #          1/2 * log(1 + sum([exp(-4), exp(-2), exp(-2)])) = 1.253856
        # Loss = (0.920522 + 1.753856 + 1.753856 + 1.253856) / 4 = 1.420522

        self.assertAllClose(loss, 1.420522)

        # Test sparse tensor
        y_true = tf.sparse.from_dense(y_true)
        loss = nml_obj(y_true, y_pred)
        self.assertAllClose(loss, 1.420522)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
