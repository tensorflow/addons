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
"""Tests for triplet loss."""

import pytest
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


def squared_l_2_dists(embs):
    return pairwise_distance_np(embs, True)


def l_2_dists(embs):
    return pairwise_distance_np(embs, False)


def angular_distance_np(feature):
    """Computes the angular distance matrix in numpy.
    Args:
      feature: 2-D numpy array of size [number of data, feature dimension]
    Returns:
      angular_distances: 2-D numpy array of size
        [number of data, number of data].
    """

    # l2-normalize all features
    normed = feature / np.linalg.norm(feature, ord=2, axis=1, keepdims=True)
    cosine_similarity = normed @ normed.T
    inverse_cos_sim = 1 - cosine_similarity

    return inverse_cos_sim


def triplet_semihard_loss_np(labels, embedding, margin, dist_func):

    num_data = embedding.shape[0]
    # Reshape labels to compute adjacency matrix.
    labels_reshaped = np.reshape(labels.astype(np.float32), (labels.shape[0], 1))

    adjacency = np.equal(labels_reshaped, labels_reshaped.T)
    pdist_matrix = dist_func(embedding)
    loss_np = 0.0
    num_positives = 0.0
    for i in range(num_data):
        for j in range(num_data):
            if adjacency[i][j] > 0.0 and i != j:
                num_positives += 1.0

                pos_distance = pdist_matrix[i][j]
                neg_distances = []

                for k in range(num_data):
                    if adjacency[i][k] == 0:
                        neg_distances.append(pdist_matrix[i][k])

                # Sort by distance.
                neg_distances.sort()
                chosen_neg_distance = neg_distances[0]

                for m in range(len(neg_distances)):
                    chosen_neg_distance = neg_distances[m]
                    if chosen_neg_distance > pos_distance:
                        break

                loss_np += np.maximum(0.0, margin - chosen_neg_distance + pos_distance)

    loss_np /= num_positives
    return loss_np


def triplet_hard_loss_np(labels, embedding, margin, dist_func, soft=False):

    num_data = embedding.shape[0]
    # Reshape labels to compute adjacency matrix.
    labels_reshaped = np.reshape(labels.astype(np.float32), (labels.shape[0], 1))

    adjacency = np.equal(labels_reshaped, labels_reshaped.T)
    pdist_matrix = dist_func(embedding)
    loss_np = 0.0
    for i in range(num_data):
        pos_distances = []
        neg_distances = []
        for j in range(num_data):
            if adjacency[i][j] == 0:
                neg_distances.append(pdist_matrix[i][j])
            if adjacency[i][j] > 0.0 and i != j:
                pos_distances.append(pdist_matrix[i][j])

        # if there are no positive pairs, distance is 0
        if len(pos_distances) == 0:
            pos_distances.append(0)

        # Sort by distance.
        neg_distances.sort()
        min_neg_distance = neg_distances[0]
        pos_distances.sort(reverse=True)
        max_pos_distance = pos_distances[0]

        if soft:
            loss_np += np.log1p(np.exp(max_pos_distance - min_neg_distance))
        else:
            loss_np += np.maximum(0.0, max_pos_distance - min_neg_distance + margin)

    loss_np /= num_data
    return loss_np


# triplet semihard
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16, tf.bfloat16])
@pytest.mark.parametrize(
    "dist_func, dist_metric",
    [
        (angular_distance_np, "angular"),
        (squared_l_2_dists, "squared-L2"),
        (l_2_dists, "L2"),
    ],
)
def test_semihard_tripled_loss_angular(dtype, dist_func, dist_metric):
    num_data = 10
    feat_dim = 6
    margin = 1.0
    num_classes = 4

    embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data))

    # Compute the loss in NP.
    loss_np = triplet_semihard_loss_np(labels, embedding, margin, dist_func)

    # Compute the loss in TF.
    y_true = tf.constant(labels)
    y_pred = tf.constant(embedding, dtype=dtype)
    cce_obj = triplet.TripletSemiHardLoss(distance_metric=dist_metric)
    loss = cce_obj(y_true, y_pred)
    test_utils.assert_allclose_according_to_type(loss.numpy(), loss_np)


def test_keras_model_compile_semihard():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Input(shape=(784,)), tf.keras.layers.Dense(10)]
    )
    model.compile(loss="Addons>triplet_semihard_loss", optimizer="adam")


def test_serialization_semihard():
    loss = triplet.TripletSemiHardLoss()
    tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))


# test cosine similarity
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16, tf.bfloat16])
@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize(
    "dist_func, dist_metric",
    [
        (angular_distance_np, "angular"),
        (squared_l_2_dists, "squared-L2"),
        (l_2_dists, "L2"),
    ],
)
def test_hard_tripled_loss_angular(dtype, soft, dist_func, dist_metric):
    num_data = 20
    feat_dim = 6
    margin = 1.0
    num_classes = 4

    embedding = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=(num_data))

    # Compute the loss in NP.
    loss_np = triplet_hard_loss_np(labels, embedding, margin, dist_func, soft)

    # Compute the loss in TF.
    y_true = tf.constant(labels)
    y_pred = tf.constant(embedding, dtype=dtype)
    cce_obj = triplet.TripletHardLoss(soft=soft, distance_metric=dist_metric)
    loss = cce_obj(y_true, y_pred)
    test_utils.assert_allclose_according_to_type(loss.numpy(), loss_np)


def test_keras_model_compile_hard():
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Input(shape=(784,)), tf.keras.layers.Dense(10)]
    )
    model.compile(loss="Addons>triplet_hard_loss", optimizer="adam")


def test_serialization_hard():
    loss = triplet.TripletHardLoss()
    tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))
