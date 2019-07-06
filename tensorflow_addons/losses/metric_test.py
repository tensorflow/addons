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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_addons.losses.metric_learning import pairwise_distance
from tensorflow_addons.utils import test_utils


def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy. Originally found in
    https://github.com/omoindrot/tensorflow-triplet-loss.

    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(
        feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
        pairwise_distances.diagonal())
    return pairwise_distances


@test_utils.run_all_in_graph_and_eager_modes
class PairWiseDistance(tf.test.TestCase):
    def test_zero_distance(self):
        """Test that equal embeddings have a pairwise distance of 0."""
        rand_embedding = tf.random.uniform([1, 3], dtype=tf.float32)
        equal_embeddings = tf.concat([rand_embedding, rand_embedding], axis=0)

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
        tf_embeddings = tf.constant([[0.7373636, 0.6229943, 0.49961114],
                                     [0.21469843, 0.26616073, 0.25712824]])

        np_embeddings = np.array([[0.7373636, 0.6229943, 0.49961114],
                                  [0.21469843, 0.26616073, 0.25712824]])

        distances = pairwise_distance(tf_embeddings, squared=False)
        self.assertAllClose(
            pairwise_distance_np(np_embeddings, squared=False), distances)

    def test_correct_distance_squared(self):
        """Compare against numpy caluclation for squared distances."""
        tf_embeddings = tf.constant([[0.7373636, 0.6229943, 0.49961114],
                                     [0.21469843, 0.26616073, 0.25712824]])

        np_embeddings = np.array([[0.7373636, 0.6229943, 0.49961114],
                                  [0.21469843, 0.26616073, 0.25712824]])

        distances = pairwise_distance(tf_embeddings, squared=True)
        self.assertAllClose(
            pairwise_distance_np(np_embeddings, squared=True), distances)


if __name__ == "__main__":
    tf.test.main()
