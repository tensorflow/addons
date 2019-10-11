## Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Conditional Random Field loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.losses.crf_loss import ConditionalRandomFieldLoss
from tensorflow_addons.metrics.crf_accuracy import ConditionalRandomFieldAccuracy
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class ConditionalRandomFieldAccuracyTest(tf.test.TestCase):
    def setUp(self):
        super(ConditionalRandomFieldAccuracyTest, self).setUp()

        self.logits = np.array([
            [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
            [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
        ])
        self.tags = np.array([[2, 3, 4], [3, 2, 2]])

        self.transitions = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])

        self.boundary_values = np.ones((5, ))

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=tf.keras.initializers.Constant(self.transitions),
            use_boundary=True,
            boundary_initializer=tf.keras.initializers.Constant(
                self.boundary_values),
            name="crf_layer",
        )

    def test_model_fit(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile(
            "adam",
            loss={"crf_layer": ConditionalRandomFieldLoss()},
            metrics=[ConditionalRandomFieldAccuracy()])

        model.fit(self.logits, self.tags, epochs=10, batch_size=1)

    def test_dump_and_load(self):
        MODEL_PERSISTENCE_PATH = './test_saving_crf_model.h5'

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile(
            "adam",
            loss={"crf_layer": ConditionalRandomFieldLoss()},
            metrics=[ConditionalRandomFieldAccuracy()])

        model.fit(self.logits, self.tags, epochs=10, batch_size=1)

        model.save(MODEL_PERSISTENCE_PATH)
        new_model = tf.keras.models.load_model(MODEL_PERSISTENCE_PATH)

        new_model.fit(self.logits, self.tags, epochs=10, batch_size=1)

        tf.keras.models.load_model(MODEL_PERSISTENCE_PATH)

        try:
            os.remove(MODEL_PERSISTENCE_PATH)
        except OSError:
            pass


if __name__ == "__main__":
    tf.test.main()
