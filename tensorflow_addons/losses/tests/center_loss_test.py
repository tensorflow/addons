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
"""Tests for center loss."""

import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.losses import CenterLoss


@test_utils.run_all_in_graph_and_eager_modes
class CenterLossTest(tf.test.TestCase):
    def test_config(self):
        center_obj = CenterLoss(
            reduction=tf.keras.losses.Reduction.NONE, name="center_loss"
        )
        self.assertEqual(center_obj.name, "center_loss")
        self.assertEqual(center_obj.reduction, tf.keras.losses.Reduction.NONE)

    def test_zero_loss(self):
        center_obj = CenterLoss()
        y_true = tf.constant([0, 0, 1, 1, 0, 1], dtype=tf.dtypes.int64)
        y_pred = tf.constant([1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=tf.dtypes.float32)
        loss = center_obj(y_true, y_pred)
        self.assertAllClose(loss, 0.0)

    def test_keras_model_compile(self):
        center_obj = CenterLoss()
        model = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(400), tf.keras.layers.Dense(10),]
        )
        model.compile(loss=center_obj, optimizer="adam")


if __name__ == "__main__":
    tf.test.main()
