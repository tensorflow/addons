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
"""Tests for GIOU loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.losses import giou_loss, GIOULoss


@test_utils.run_all_in_graph_and_eager_modes
class GIOULossTest(tf.test.TestCase):
    """GIOU test class."""

    def test_config(self):
        gl_obj = GIOULoss(
            reduction=tf.keras.losses.Reduction.NONE, name='giou_loss')
        self.assertEqual(gl_obj.name, 'giou_loss')
        self.assertEqual(gl_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_giou_loss(self):
        pred_data = [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 2, 2]]
        true_data = [[0, 0, 1, 1], [0, 0, 1, 2], [2, 2, 4, 4]]
        loss = giou_loss(true_data, pred_data)
        self.assertAllClose(loss, tf.constant([0, 0.5, 1.25], tf.float64))


if __name__ == '__main__':
    tf.test.main()
