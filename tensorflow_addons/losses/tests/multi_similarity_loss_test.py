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
"""Tests for Multi Similarity Loss."""


import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.losses import multi_similarity_loss, MultiSimilarityLoss


def test_config():
    bce_obj = MultiSimilarityLoss(name="multi_similarity_loss")
    assert bce_obj.name == "multi_similarity_loss"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ms_loss_np(
    y_true,
    y_pred,
    alpha=2.0,
    beta=2.0,
    lamb=1.0,
    eps=0.1,
    ms_mining=False,
    from_logits=False,
):
    if from_logits:
        y_pred = sigmoid(y_pred)
    y_true = np.reshape(y_true, (-1, 1))
    y_pred = np.reshape(y_pred, (-1, 1))
    batch_size = y_true.size
    adjacency = np.equal(y_true, np.transpose(y_true))
    adjacency_not = np.logical_not(adjacency)
    mask_pos = adjacency.astype(np.float32) - np.eye(batch_size, dtype=np.float32)
    mask_neg = adjacency_not.astype(np.float32)
    sim_mat = np.matmul(y_pred, np.transpose(y_pred))
    pos_mat = np.multiply(sim_mat, mask_pos)
    neg_mat = np.multiply(sim_mat, mask_neg)
    if ms_mining:
        max_val = np.amax(neg_mat, axis=1, keepdims=True)
        tmp_max_val = np.amax(pos_mat, axis=1, keepdims=True)
        min_val = (
            np.amin(np.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True)
            + tmp_max_val
        )
        max_val = np.tile(max_val, (1, batch_size))
        min_val = np.tile(min_val, (1, batch_size))
        mask_pos = np.where(pos_mat < max_val + eps, mask_pos, np.zeros_like(mask_pos))
        mask_neg = np.where(neg_mat > min_val - eps, mask_neg, np.zeros_like(mask_neg))
    pos_exp = np.exp(-alpha * (pos_mat - lamb))
    pos_exp = np.where(mask_pos > 0.0, pos_exp, np.zeros_like(pos_exp))
    neg_exp = np.exp(beta * (neg_mat - lamb))
    neg_exp = np.where(mask_neg > 0.0, neg_exp, np.zeros_like(neg_exp))
    pos_term = np.log(1.0 + np.sum(pos_exp, axis=1)) / alpha
    neg_term = np.log(1.0 + np.sum(neg_exp, axis=1)) / beta
    loss = np.mean(pos_term + neg_term)
    return loss


def get_labels_and_preds(out_shape=(2, 3)):
    y_true = np.array(np.random.uniform(size=(2, 3)), dtype=np.float32)
    eps = np.finfo(float).eps
    y_pred = y_true + eps
    return y_true, y_pred


@pytest.mark.parametrize("out_shape", [(2, 3), (1, 8), (8, 1)])
def test_ms_loss_for_shapes(out_shape):
    y_true, y_pred = get_labels_and_preds(out_shape)
    np_loss = ms_loss_np(y_true, y_pred)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0, 10.0])
def test_ms_loss_for_alpha(alpha):
    y_true, y_pred = get_labels_and_preds()
    np_loss = ms_loss_np(y_true, y_pred, alpha=alpha)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), alpha=alpha
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


@pytest.mark.parametrize("beta", [1.0, 2.0, 4.0, 10.0])
def test_ms_loss_for_beta(beta):
    y_true, y_pred = get_labels_and_preds()
    np_loss = ms_loss_np(y_true, y_pred, beta=beta)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), beta=beta
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


@pytest.mark.parametrize("lamb", [2.0, 4.0, 8.0, 10.0])
def test_ms_loss_for_lamb(lamb):
    y_true, y_pred = get_labels_and_preds()
    np_loss = ms_loss_np(y_true, y_pred, lamb=lamb)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), lamb=lamb
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-7])
def test_ms_loss_for_eps(eps):
    y_true, y_pred = get_labels_and_preds()
    np_loss = ms_loss_np(y_true, y_pred, eps=eps)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), eps=eps
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


@pytest.mark.parametrize("ms_mining", [False, True])
def test_ms_loss_for_mining(ms_mining):
    y_true, y_pred = get_labels_and_preds()
    np_loss = ms_loss_np(y_true, y_pred, ms_mining=ms_mining)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred), ms_mining=ms_mining
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


@pytest.mark.parametrize("from_logits", [False, True])
def test_ms_loss_for_logits(from_logits):
    y_true, y_pred = get_labels_and_preds()
    np_loss = ms_loss_np(y_true, y_pred, from_logits=from_logits)
    tf_loss = multi_similarity_loss(
        tf.convert_to_tensor(y_true),
        tf.convert_to_tensor(y_pred),
        from_logits=from_logits,
    ).numpy()
    np.testing.assert_almost_equal(np_loss, tf_loss, decimal=5)


def test_serialization():
    loss = MultiSimilarityLoss()
    tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))
