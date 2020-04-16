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
"""Implements Multi Similarity loss."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultiSimilarityLoss(tf.keras.losses.Loss):
    """Implements the multi similarity loss function.

    Multi Similarity Loss was first introduced in "Multi-Similarity Loss with
    General Pair Weighting for Deep Metric Learning" (https://arxiv.org/pdf/1904.06627.pdf).
    This loss is implemented in two iterative steps (i.e., mining and weighting).
    This allows it to fully consider three similarities for pair weighting,
    providing a more principled approach for collecting and weighting informative pairs.
    Finally, the proposed MS loss obtains new state-of-the-art performance on four image
    retrieval benchmarks, where it outperforms the most recent approaches.

    Usage:

    ```python
    msl = tfa.losses.MultiSimilarityLoss()
    loss = msl(
      [[0.97], [0.91], [0.03]],
      [[1.0], [1.0], [0.0]])
    print('Loss: ', loss.numpy())  # Loss:  0.009241962
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.MultiSimilarityLoss())
    ```

    Args:
      alpha: Hyper-parameter similar as Binomial Deviance Loss.
      beta: Hyper-parameter similar as Binomial Deviance Loss.
      lamb: Hyper-parameter similar as Binomial Deviance Loss.
      eps: very small positive number to prevent devide by zero error.
      ms_mining: bool.

    Returns:
      Loss float `Tensor`.
    """

    @typechecked
    def __init__(
        self,
        alpha: FloatTensorLike = 2.0,
        beta: FloatTensorLike = 50.0,
        lamb: FloatTensorLike = 1.0,
        eps: FloatTensorLike = 0.1,
        ms_mining: bool = False,
        from_logits: bool = False,
        name: str = "multi_similarity_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        self.eps = eps
        self.ms_mining = ms_mining
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return multi_similarity_loss(
            y_true,
            y_pred,
            self.alpha,
            self.beta,
            self.lamb,
            self.eps,
            self.ms_mining,
            self.from_logits,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def multi_similarity_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 2.0,
    beta: FloatTensorLike = 50.0,
    lamb: FloatTensorLike = 1.0,
    eps: FloatTensorLike = 0.1,
    ms_mining: bool = False,
    from_logits: bool = False,
):
    """Implements the multi similarity loss function.

    Multi Similarity Loss was first introduced in "Multi-Similarity Loss with
    General Pair Weighting for Deep Metric Learning" (https://arxiv.org/pdf/1904.06627.pdf).
    This loss is implemented in two iterative steps (i.e., mining and weighting).
    This allows it to fully consider three similarities for pair weighting,
    providing a more principled approach for collecting and weighting informative pairs.
    Finally, the proposed MS loss obtains new state-of-the-art performance on four image
    retrieval benchmarks, where it outperforms the most recent approaches.

    Usage:

    ```python
    msl = tfa.losses.MultiSimilarityLoss()
    loss = msl(
      [[0.97], [0.91], [0.03]],
      [[1.0], [1.0], [0.0]])
    print('Loss: ', loss.numpy())
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.MultiSimilarityLoss())
    ```

    Args:
      alpha: Hyper-parameter similar as Binomial Deviance Loss.
      beta: Hyper-parameter similar as Binomial Deviance Loss.
      lamb: Hyper-parameter similar as Binomial Deviance Loss.
      eps: very small positive number to prevent devide by zero error.
      ms_mining: bool.

    Returns:
      Loss float `Tensor`.
    """
    if from_logits:
        y_pred = tf.sigmoid(y_pred)
    y_true = tf.reshape(y_true, [-1, 1])
    batch_size = tf.size(y_true)
    adjacency = tf.equal(y_true, tf.transpose(y_true))
    adjacency_not = tf.logical_not(adjacency)
    mask_pos = tf.cast(adjacency, dtype=tf.float32) - tf.eye(
        batch_size, dtype=tf.float32
    )
    mask_neg = tf.cast(adjacency_not, dtype=tf.float32)

    sim_mat = tf.matmul(y_pred, y_pred, transpose_a=False, transpose_b=True)
    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = (
            tf.reduce_min(
                tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True
            )
            + tmp_max_val
        )

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.math.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.math.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term)
    return loss
