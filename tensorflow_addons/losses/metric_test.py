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
"""Tests for metric learning."""

import numpy as np
import tensorflow as tf
from tensorflow_addons.losses.metric_learning import pairwise_distance
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class PairWiseDistance(tf.test.TestCase):
    def test_zero_distance(self):
        """Test that equal embeddings have a pairwise distance of 0."""
        equal_embeddings = tf.constant([[1.0, 0.5], [1.0, 0.5]])

        distances = pairwise_distance(equal_embeddings, squared=False)
        self.assertAllClose(tf.math.reduce_sum(distances), 0)

    def test_positive_distances(self):
        """Test that the pairwise distances are always positive."""

        # Create embeddings very close to each other in [1.0 - 2e-7, 1.0 + 2e-7]
        # This will encourage errors in the computation
        embeddings = 1.0 + 2e-7 * tf.random.uniform([64, 6], dtype=tf.float32)
        distances = pairwise_distance(embeddings, squared=False)
        self.assertAllGreaterEqual(distances, 0)

    def test_correct_distance(self):
        """Compare against numpy caluclation."""
        tf_embeddings = tf.constant([[0.5, 0.5], [1.0, 1.0]])

        expected_distance = np.array([[0, np.sqrt(2) / 2], [np.sqrt(2) / 2,
                                                            0]])

        distances = pairwise_distance(tf_embeddings, squared=False)
        self.assertAllClose(expected_distance, distances)

    def test_correct_distance_squared(self):
        """Compare against numpy caluclation for squared distances."""
        tf_embeddings = tf.constant([[0.5, 0.5], [1.0, 1.0]])

        expected_distance = np.array([[0, 0.5], [0.5, 0]])

        distances = pairwise_distance(tf_embeddings, squared=True)
        self.assertAllClose(expected_distance, distances)


if __name__ == "__main__":
    tf.test.main()
