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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.losses import npairs
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class NpairsLossTest(tf.test.TestCase):
    def test_config(self):
        nl_obj = npairs.NpairsLoss(name="nl")
        self.assertEqual(nl_obj.name, "nl")
        self.assertEqual(nl_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_unweighted(self):
        nl_obj = npairs.NpairsLoss()
        y_true = tf.constant([0, 0, 1, 1, 2], dtype=tf.int64)
        y_pred = tf.constant(
            [[0.0, 0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9],
             [1.0, 1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8, 1.9],
             [2.0, 2.1, 2.2, 2.3, 2.4]],
            dtype=tf.float32)
        loss = nl_obj(y_true, y_pred)
        self.assertAllClose(loss, 1.619416)


if __name__ == "__main__":
    tf.test.main()
