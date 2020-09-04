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
"""Implements npairs loss."""

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils.types import TensorLike


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def npairs_loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    """Computes the npairs loss between `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from
    the same labels and each pairs in the minibatch have different labels.
    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b`
    with shape `[batch_size, hidden_size]` can be computed as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
    >>> y_pred
    <tf.Tensor: shape=(3, 3), dtype=float16, numpy=
    array([[23., 15., 17.],
       [51., 33., 35.],
       [79., 51., 53.]], dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      y_true: 1-D integer `Tensor` with shape `[batch_size]` of
        multi-class labels.
      y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
        similarity matrix between embedding matrices.

    Returns:
      npairs_loss: float scalar.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Expand to [batch_size, 1]
    y_true = tf.expand_dims(y_true, -1)
    y_true = tf.cast(tf.equal(y_true, tf.transpose(y_true)), y_pred.dtype)
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    return tf.math.reduce_mean(loss)


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def npairs_multilabel_loss(y_true: TensorLike, y_pred: TensorLike) -> tf.Tensor:
    r"""Computes the npairs loss between multilabel data `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from
    the same labels and each pairs in the minibatch have different labels.
    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    To deal with multilabel inputs, the count of label intersection
    is computed as follows:

    ```
    L_{i,j} = | set_of_labels_for(i) \cap set_of_labels_for(j) |
    ```

    Each row of the count based label matrix is further normalized so that
    each row sums to one.

    `y_true` should be a binary indicator for classes.
    That is, if `y_true[i, j] = 1`, then `i`th sample is in `j`th class;
    if `y_true[i, j] = 0`, then `i`th sample is not in `j`th class.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b`
    with shape `[batch_size, hidden_size]` can be computed as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
    >>> y_pred
    <tf.Tensor: shape=(3, 3), dtype=float16, numpy=
    array([[23., 15., 17.],
       [51., 33., 35.],
       [79., 51., 53.]], dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      y_true: Either 2-D integer `Tensor` with shape
        `[batch_size, num_classes]`, or `SparseTensor` with dense shape
        `[batch_size, num_classes]`. If `y_true` is a `SparseTensor`, then
        it will be converted to `Tensor` via `tf.sparse.to_dense` first.

      y_pred: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
        similarity matrix between embedding matrices.

    Returns:
      npairs_multilabel_loss: float scalar.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Convert to dense tensor if `y_true` is a `SparseTensor`
    if isinstance(y_true, tf.SparseTensor):
        y_true = tf.sparse.to_dense(y_true)

    # Enable efficient multiplication because y_true contains lots of zeros
    # https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
    y_true = tf.linalg.matmul(
        y_true, y_true, transpose_b=True, a_is_sparse=True, b_is_sparse=True
    )
    y_true /= tf.math.reduce_sum(y_true, 1, keepdims=True)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

    return tf.math.reduce_mean(loss)


@tf.keras.utils.register_keras_serializable(package="Addons")
class NpairsLoss(tf.keras.losses.Loss):
    """Computes the npairs loss between `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from
    the same labels and each pairs in the minibatch have different labels.
    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b`
    with shape `[batch_size, hidden_size]` can be computed as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
    >>> y_pred
    <tf.Tensor: shape=(3, 3), dtype=float16, numpy=
    array([[23., 15., 17.],
       [51., 33., 35.],
       [79., 51., 53.]], dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      name: (Optional) name for the loss.
    """

    @typechecked
    def __init__(self, name: str = "npairs_loss"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred):
        return npairs_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="Addons")
class NpairsMultilabelLoss(tf.keras.losses.Loss):
    r"""Computes the npairs loss between multilabel data `y_true` and `y_pred`.

    Npairs loss expects paired data where a pair is composed of samples from
    the same labels and each pairs in the minibatch have different labels.
    The loss takes each row of the pair-wise similarity matrix, `y_pred`,
    as logits and the remapped multi-class labels, `y_true`, as labels.

    To deal with multilabel inputs, the count of label intersection
    is computed as follows:

    ```
    L_{i,j} = | set_of_labels_for(i) \cap set_of_labels_for(j) |
    ```

    Each row of the count based label matrix is further normalized so that
    each row sums to one.

    `y_true` should be a binary indicator for classes.
    That is, if `y_true[i, j] = 1`, then `i`th sample is in `j`th class;
    if `y_true[i, j] = 0`, then `i`th sample is not in `j`th class.

    The similarity matrix `y_pred` between two embedding matrices `a` and `b`
    with shape `[batch_size, hidden_size]` can be computed as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],
    ...                 [1, 8]], dtype=tf.float16)
    >>> y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
    >>> y_pred
    <tf.Tensor: shape=(3, 3), dtype=float16, numpy=
    array([[23., 15., 17.],
       [51., 33., 35.],
       [79., 51., 53.]], dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

    Args:
      name: (Optional) name for the loss.
    """

    @typechecked
    def __init__(self, name: str = "npairs_multilabel_loss"):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

    def call(self, y_true, y_pred):
        return npairs_multilabel_loss(y_true, y_pred)
