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
"""Tests for triplet hard loss."""

import numpy as np
import tensorflow as tf

from tensorflow_addons.losses import triplet
from tensorflow_addons.utils import test_utils

def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.

    Args:
      feature: 2-D numpy array of size [number of data, feature dimension]
      squared: Boolean. If true, output is the pairwise squared euclidean
        distance matrix; else, output is the pairwise euclidean distance
        matrix.

    Returns:
      pairwise_distances: 2-D numpy array of size
        [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.0
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = (
        pairwise_distances
        + pairwise_distances.T
        - np.diag(pairwise_distances.diagonal())
    )
    return pairwise_distances

@test_utils.run_all_in_graph_and_eager_modes
class TripletHardLossTest(tf.test.TestCase):
    def test_unweighted(self):
        num_data = 20
        feat_dim = 6
        margin = 1.0
        num_classes = 4

        embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=(num_data))

        # Reshape labels to compute adjacency matrix.
        labels_reshaped = np.reshape(
            labels.astype(np.float32), (labels.shape[0], 1))
        # Compute the loss in NP.
        adjacency = np.equal(labels_reshaped, labels_reshaped.T)

        pdist_matrix = pairwise_distance_np(embedding, squared=True)
        loss_np = 0.0
        for i in range(num_data):
            pos_distances = []
            neg_distances = []
            for j in range(num_data):
                if adjacency[i][j] == 0:
                    neg_distances.append(pdist_matrix[i][j])
                if adjacency[i][j] > 0.0 and i != j:
                    pos_distances.append(pdist_matrix[i][j])

            # if their are no positive pairs, distance is 0
            if len(pos_distances) == 0:
                pos_distances.append(0)

            # Sort by distance.
            neg_distances.sort()
            min_neg_distance = neg_distances[0]
            pos_distances.sort(reverse=True)
            max_pos_distance = pos_distances[0]

            loss_np += np.maximum(
                0.0, margin - min_neg_distance + max_pos_distance)

        loss_np /= num_data

        # Compute the loss in TF.
        y_true = tf.constant(labels)
        y_pred = tf.constant(embedding)
        cce_obj = triplet.TripletHardLoss()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), loss_np, 3)

    def test_keras_model_compile(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(10),
        ])
        model.compile(loss="Addons>triplet_hard_loss", optimizer="adam")

    def test_serialization(self):
        loss = triplet.TripletHardLoss()
        new_loss = tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))


if __name__ == '__main__':
    tf.test.main()
