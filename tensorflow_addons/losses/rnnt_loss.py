# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Implements RNN-T loss."""

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import matrix_diag_part_v2
from typeguard import typechecked

from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import TensorLike

LOG_0 = float("-inf")


@tf.keras.utils.register_keras_serializable(package="Addons")
class RNNTLoss(LossFunctionWrapper):
    """Implements RNN-T loss function.

    RNN-T Loss was first introduced in Sequence Transduction
    with Recurrent Neural Networks (https://arxiv.org/abs/1211.3711).
    This loss function is commonly used to train Automatic Speech
    Recognition (ASR) models.

    Usage:
    >>> import tensorflow_addons as tfa

    >>> loss = tfa.losses.RNNTLoss()

    Args:
        reduction: reduction method from tf.keras.losses.Reduction
        name: name of function

    Returns:
        loss float 'Tensor'. If 'reduction' is 'NONE', this has the same shape as 'y_true'; otherwise it is a scalar.
    """

    @typechecked
    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "rnnt_loss",
    ):
        super().__init__(
            rnnt_loss,
            name=name,
            reduction=reduction,
            from_logits=True,
        )


@tf.keras.utils.register_keras_serializable(package="Addons")
@tf.function
def rnnt_loss(
    logits: TensorLike,
    labels: TensorLike,
    label_length: TensorLike,
    logit_length: TensorLike,
    name: str = "rnnt_loss",
) -> TensorLike:
    with tf.name_scope(name):
        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        label_length = tf.convert_to_tensor(label_length, name="label_length")
        logit_length = tf.convert_to_tensor(logit_length, name="logit_length")

        args = [logits, labels, label_length, logit_length]

        @tf.custom_gradient
        def _compute_rnnt_loss_and_grad(
            logits_t, labels_t, label_length_t, logit_length_t
        ):
            """Compute RNN-T loss and gradients."""
            logits_t.set_shape(logits.shape)
            labels_t.set_shape(labels.shape)
            label_length_t.set_shape(label_length.shape)
            logit_length_t.set_shape(logit_length.shape)
            kwargs = dict(
                logits=logits_t,
                labels=labels_t,
                label_length=label_length_t,
                logit_length=logit_length_t,
            )
            result = _compute_rnnt_loss_and_grad_helper(**kwargs)

            def _grad(grad_loss):
                grads = [tf.reshape(grad_loss, [-1, 1, 1, 1]) * result[1]]
                grads += [None] * (len(args) - len(grads))
                return grads

            return result[0], _grad

        return _compute_rnnt_loss_and_grad(*args)


def _nan_to_zero(input_tensor):
    return tf.where(
        tf.math.is_nan(input_tensor), tf.zeros_like(input_tensor), input_tensor
    )


def _reduce_logsumexp(input_tensor, axis):
    maximum = tf.reduce_max(input_tensor, axis=axis)
    input_tensor = _nan_to_zero(input_tensor - maximum)
    return tf.math.log(tf.reduce_sum(tf.exp(input_tensor), axis=axis)) + maximum


def _extract_diagonals(log_probs):
    time_steps = tf.shape(log_probs)[1]  # T
    output_steps = tf.shape(log_probs)[2]  # U + 1
    reverse_log_probs = tf.reverse(log_probs, axis=[-1])
    paddings = [[0, 0], [0, 0], [time_steps - 1, 0]]
    padded_reverse_log_probs = tf.pad(
        reverse_log_probs, paddings, "CONSTANT", constant_values=LOG_0
    )
    diagonals = matrix_diag_part_v2(
        padded_reverse_log_probs,
        k=(0, time_steps + output_steps - 2),
        padding_value=LOG_0,
    )

    return tf.transpose(diagonals, perm=[1, 0, 2])


def _transition_probs(one_hot_labels, log_probs):
    blank_probs = log_probs[:, :, :, 0]
    truth_probs = tf.reduce_sum(
        tf.multiply(log_probs[:, :, :-1, :], one_hot_labels), axis=-1
    )

    return blank_probs, truth_probs


def _forward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len):
    def _next_state(x, trans_probs):
        blank_probs = trans_probs[0]
        truth_probs = trans_probs[1]

        x_b = tf.concat(
            [LOG_0 * tf.ones(shape=[batch_size, 1]), x[:, :-1] + blank_probs], axis=1
        )
        x_t = x + truth_probs

        x = tf.math.reduce_logsumexp(tf.stack([x_b, x_t], axis=0), axis=0)
        return x

    initial_alpha = tf.concat(
        [
            tf.zeros(shape=[batch_size, 1]),
            tf.ones(shape=[batch_size, input_max_len - 1]) * LOG_0,
        ],
        axis=1,
    )

    fwd = tf.scan(
        _next_state, (bp_diags[:-1, :, :-1], tp_diags), initializer=initial_alpha
    )

    alpha = tf.transpose(
        tf.concat([tf.expand_dims(initial_alpha, axis=0), fwd], axis=0), perm=[1, 2, 0]
    )
    alpha = matrix_diag_part_v2(alpha, k=(0, target_max_len - 1), padding_value=LOG_0)
    alpha = tf.transpose(tf.reverse(alpha, axis=[1]), perm=[0, 2, 1])

    return alpha


def _backward_dp(
    bp_diags,
    tp_diags,
    batch_size,
    input_max_len,
    target_max_len,
    label_length,
    logit_length,
    blank_sl,
):
    def _next_state(x, mask_and_trans_probs):
        mask_s, blank_probs_s, truth_probs = mask_and_trans_probs

        beta_b = tf.concat(
            [x[:, 1:] + blank_probs_s, LOG_0 * tf.ones(shape=[batch_size, 1])], axis=1
        )
        beta_t = tf.concat(
            [x[:, :-1] + truth_probs, LOG_0 * tf.ones(shape=[batch_size, 1])], axis=1
        )

        beta_next = _reduce_logsumexp(tf.stack([beta_b, beta_t], axis=0), axis=0)
        masked_beta_next = _nan_to_zero(
            beta_next * tf.expand_dims(mask_s, axis=1)
        ) + _nan_to_zero(x * tf.expand_dims((1.0 - mask_s), axis=1))
        return masked_beta_next

    # Initial beta for batches.
    initial_beta_mask = tf.one_hot(logit_length - 1, depth=input_max_len + 1)
    initial_beta = tf.expand_dims(blank_sl, axis=1) * initial_beta_mask + _nan_to_zero(
        LOG_0 * (1.0 - initial_beta_mask)
    )

    # Mask for scan iterations.
    mask = tf.sequence_mask(
        logit_length + label_length - 1,
        input_max_len + target_max_len - 2,
        dtype=tf.dtypes.float32,
    )
    mask = tf.transpose(mask, perm=[1, 0])

    bwd = tf.scan(
        _next_state,
        (mask, bp_diags[:-1, :, :], tp_diags),
        initializer=initial_beta,
        reverse=True,
    )

    beta = tf.transpose(
        tf.concat([bwd, tf.expand_dims(initial_beta, axis=0)], axis=0), perm=[1, 2, 0]
    )[:, :-1, :]
    beta = matrix_diag_part_v2(beta, k=(0, target_max_len - 1), padding_value=LOG_0)
    beta = tf.transpose(tf.reverse(beta, axis=[1]), perm=[0, 2, 1])

    return beta


def _compute_rnnt_loss_and_grad_helper(logits, labels, label_length, logit_length):
    batch_size = logits.shape[0]
    input_max_len = logits.shape[1]
    target_max_len = logits.shape[2]
    vocab_size = logits.shape[3]

    one_hot_labels = tf.one_hot(
        tf.tile(tf.expand_dims(labels, axis=1), multiples=[1, input_max_len, 1]),
        depth=vocab_size,
    )

    log_probs = tf.nn.log_softmax(logits)
    blank_probs, truth_probs = _transition_probs(one_hot_labels, log_probs)
    bp_diags = _extract_diagonals(blank_probs)
    tp_diags = _extract_diagonals(truth_probs)

    label_mask = tf.expand_dims(
        tf.sequence_mask(label_length + 1, maxlen=target_max_len, dtype=tf.float32),
        axis=1,
    )
    small_label_mask = tf.expand_dims(
        tf.sequence_mask(label_length, maxlen=target_max_len, dtype=tf.float32), axis=1
    )
    input_mask = tf.expand_dims(
        tf.sequence_mask(logit_length, maxlen=input_max_len, dtype=tf.float32), axis=2
    )
    small_input_mask = tf.expand_dims(
        tf.sequence_mask(logit_length - 1, maxlen=input_max_len, dtype=tf.float32),
        axis=2,
    )
    mask = label_mask * input_mask
    grad_blank_mask = (label_mask * small_input_mask)[:, :-1, :]
    grad_truth_mask = (small_label_mask * input_mask)[:, :, :-1]

    alpha = (
        _forward_dp(bp_diags, tp_diags, batch_size, input_max_len, target_max_len)
        * mask
    )

    indices = tf.stack([logit_length - 1, label_length], axis=1)
    blank_sl = tf.gather_nd(blank_probs, indices, batch_dims=1)

    beta = (
        _backward_dp(
            bp_diags,
            tp_diags,
            batch_size,
            input_max_len,
            target_max_len,
            label_length,
            logit_length,
            blank_sl,
        )
        * mask
    )
    beta = tf.where(tf.math.is_nan(beta), tf.zeros_like(beta), beta)
    final_state_probs = beta[:, 0, 0]

    # Compute gradients of loss w.r.t. blank log-probabilities.
    grads_blank = (
        -tf.exp(
            (
                alpha[:, :-1, :]
                + beta[:, 1:, :]
                - tf.reshape(final_state_probs, shape=[batch_size, 1, 1])
                + blank_probs[:, :-1, :]
            )
            * grad_blank_mask
        )
        * grad_blank_mask
    )
    grads_blank = tf.concat(
        [grads_blank, tf.zeros(shape=(batch_size, 1, target_max_len))], axis=1
    )
    last_grads_blank = -1 * tf.scatter_nd(
        tf.concat(
            [
                tf.reshape(tf.range(batch_size, dtype=tf.int64), shape=[batch_size, 1]),
                indices,
            ],
            axis=1,
        ),
        tf.ones(batch_size, dtype=tf.float32),
        [batch_size, input_max_len, target_max_len],
    )
    grads_blank = grads_blank + last_grads_blank

    # Compute gradients of loss w.r.t. truth log-probabilities.
    grads_truth = (
        -tf.exp(
            (
                alpha[:, :, :-1]
                + beta[:, :, 1:]
                - tf.reshape(final_state_probs, shape=[batch_size, 1, 1])
                + truth_probs
            )
            * grad_truth_mask
        )
        * grad_truth_mask
    )

    # Compute gradients of loss w.r.t. activations.
    a = tf.tile(
        tf.reshape(
            tf.range(target_max_len - 1, dtype=tf.int64),
            shape=(1, 1, target_max_len - 1, 1),
        ),
        multiples=[batch_size, 1, 1, 1],
    )
    b = tf.reshape(labels - 1, shape=(batch_size, 1, target_max_len - 1, 1))
    c = tf.concat([a, b], axis=3)
    d = tf.tile(c, multiples=(1, input_max_len, 1, 1))
    e = tf.tile(
        tf.reshape(
            tf.range(input_max_len, dtype=tf.int64), shape=(1, input_max_len, 1, 1)
        ),
        multiples=(batch_size, 1, target_max_len - 1, 1),
    )
    f = tf.concat([e, d], axis=3)
    g = tf.tile(
        tf.reshape(tf.range(batch_size, dtype=tf.int64), shape=(batch_size, 1, 1, 1)),
        multiples=[1, input_max_len, target_max_len - 1, 1],
    )
    scatter_idx = tf.concat([g, f], axis=3)
    # TODO - improve the part of code for scatter_idx computation.
    probs = tf.exp(log_probs)
    grads_truth_scatter = tf.scatter_nd(
        scatter_idx,
        grads_truth,
        [batch_size, input_max_len, target_max_len, vocab_size - 1],
    )
    grads = tf.concat(
        [
            tf.reshape(
                grads_blank, shape=(batch_size, input_max_len, target_max_len, -1)
            ),
            grads_truth_scatter,
        ],
        axis=3,
    )
    grads_logits = grads - probs * (tf.reduce_sum(grads, axis=3, keepdims=True))

    loss = -final_state_probs
    return loss, grads_logits
