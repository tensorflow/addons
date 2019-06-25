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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


@tf.function
def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """Computes the unnormalized score for a tag sequence.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """

    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of the single tag.
    def _single_seq_fn():
        batch_size = tf.shape(inputs, out_type=tag_indices.dtype)[0]

        example_inds = tf.reshape(
            tf.range(batch_size, dtype=tag_indices.dtype), [-1, 1])
        sequence_scores = tf.gather_nd(
            tf.squeeze(inputs, [1]),
            tf.concat([example_inds, tag_indices], axis=1))
        sequence_scores = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(sequence_scores),
            sequence_scores)
        return sequence_scores

    def _multi_seq_fn():
        # Compute the scores of the given tag sequence.
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                         transition_params)
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


@tf.function
def crf_multitag_sequence_score(inputs, tag_bitmap, sequence_lengths,
                                transition_params):
    """Computes the unnormalized score of all tag sequences matching
    tag_bitmap.

    tag_bitmap enables more than one tag to be considered correct at each time
    step. This is useful when an observed output at a given time step is
    consistent with more than one tag, and thus the log likelihood of that
    observation must take into account all possible consistent tags.

    Using one-hot vectors in tag_bitmap gives results identical to
    crf_sequence_score.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
          representing all active tags at each index for which to calculate the
          unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """

    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of all active tags.
    def _single_seq_fn():
        filtered_inputs = tf.where(tag_bitmap, inputs,
                                   tf.fill(tf.shape(inputs), float("-inf")))
        return tf.reduce_logsumexp(
            filtered_inputs, axis=[1, 2], keepdims=False)

    def _multi_seq_fn():
        # Compute the logsumexp of all scores of sequences matching the given tags.
        filtered_inputs = tf.where(tag_bitmap, inputs,
                                   tf.fill(tf.shape(inputs), float("-inf")))
        return crf_log_norm(
            inputs=filtered_inputs,
            sequence_lengths=sequence_lengths,
            transition_params=transition_params)

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


@tf.function
def crf_log_norm(inputs, sequence_lengths, transition_params):
    """Computes the normalization for a CRF.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # Split up the first and rest of the inputs in preparation for the forward
    # algorithm.
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
    # the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm),
            log_norm)
        return log_norm

    def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.
        forward_cell = CrfForwardRnnCell(transition_params)
        # Sequence length is not allowed to be less than zero.
        sequence_lengths_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1)

        forward_layer = tf.keras.layers.RNN(
            forward_cell, return_sequences=True, return_state=True)

        mask = tf.sequence_mask(sequence_lengths_less_one,
                                tf.shape(inputs)[1] - 1)
        _, alphas = forward_layer(rest_of_input, first_input, mask=mask)
        log_norm = tf.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm),
            log_norm)
        return log_norm

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


@tf.function
def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
    """Computes the log-likelihood of tag sequences in a CRF.

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the log-likelihood.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix,
          if available.
    Returns:
      log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
      transition_params: A [num_tags, num_tags] transition matrix. This is
          either provided by the caller or created in this function.
    """
    # Get shape information.
    num_tags = inputs.shape[2]

    # Get the transition matrix if not provided.
    if transition_params is None:
        transition_params = tf.get_variable("transitions",
                                            [num_tags, num_tags])

    sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths,
                                         transition_params)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood per example.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


@tf.function
def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """Computes the unary scores of tag sequences.

    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
      unary_scores: A [batch_size] vector of unary scores.
    """
    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
    # Use int32 or int64 based on tag_indices' dtype.
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


@tf.function
def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """Computes the binary scores of tag sequences.

    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      binary_scores: A [batch_size] vector of binary scores.
    """
    # Get shape information.
    num_tags = tf.shape(transition_params)[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    # Encode the indices in a flattened representation.
    flattened_transition_indices = start_tag_indices * \
        num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = tf.gather(flattened_transition_params,
                              flattened_transition_indices)

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


class CrfForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    """Computes the alpha values in a linear-chain CRF.

    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """

    def __init__(self, transition_params, **kwargs):
        """Initialize the CrfForwardRnnCell.
        Args:
            transition_params: A [num_tags, num_tags] matrix of binary 
                potentials. This matrix is expanded into a 
                [1, num_tags, num_tags] in preparation for the 
                broadcast summation occurring within the cell.
        """
        super(CrfForwardRnnCell, self).__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super(CrfForwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        """Build the CrfForwardRnnCell.

        Args:
            inputs: A [batch_size, num_tags] matrix of unary potentials.
            state: A [batch_size, num_tags] matrix containing the
                previous alpha values.
            scope: Unused variable scope of this cell.
            Returns:
            new_alphas, new_alphas: A pair of [batch_size, num_tags]
                matrices values containing the new alpha values.
        """
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas, new_alphas


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    """Computes the forward decoding in a linear-chain CRF."""

    def __init__(self, transition_params, **kwargs):
        """Initialize the CrfDecodeForwardRnnCell.

        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        super(CrfDecodeForwardRnnCell, self).__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super(CrfDecodeForwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_state = inputs + tf.reduce_max(transition_scores, [1])
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)
        return backpointers, new_state


class CrfDecodeBackwardRnnCell(tf.keras.layers.Layer):
    """Computes backward decoding in a linear-chain CRF."""

    def __init__(self, num_tags, **kwargs):
        """Initialize the CrfDecodeBackwardRnnCell.

        Args:
          num_tags: An integer. The number of tags.
        """
        super(CrfDecodeBackwardRnnCell, self).__init__(**kwargs)
        self._num_tags = num_tags

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def build(self, input_shape):
        super(CrfDecodeBackwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        state = tf.squeeze(state[0], axis=[1])
        batch_size = tf.shape(inputs)[0]
        b_indices = tf.range(batch_size)
        indices = tf.stack([b_indices, state], axis=1)
        new_tags = tf.expand_dims(tf.gather_nd(inputs, indices), axis=-1)

        return new_tags, new_tags


@tf.function
def crf_decode(potentials, transition_params, sequence_length):
    """Decode the highest scoring sequence of tags in TensorFlow.

    This is a function for tensor.

    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.

    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """

    # If max_seq_len is 1, we skip the algorithm and simply return the argmax tag
    # and the max activation.
    def _single_seq_fn():
        squeezed_potentials = tf.squeeze(potentials, [1])
        decode_tags = tf.expand_dims(tf.argmax(squeezed_potentials, axis=1), 1)
        best_score = tf.reduce_max(squeezed_potentials, axis=1)
        return tf.cast(decode_tags, dtype=tf.int32), best_score

    def _multi_seq_fn():
        """Decoding of highest scoring sequence."""

        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        num_tags = potentials.shape[2]

        # Computes forward decoding. Get last score and backpointers.
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])  # [B, O]
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
        # Sequence length is not allowed to be less than zero.

        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_length.dtype), sequence_length - 1)

        mask = tf.sequence_mask(sequence_length_less_one, tf.shape(inputs)[1])
        crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
        crf_fwd_layer = tf.keras.layers.RNN(
            crf_fwd_cell,
            return_sequences=True,
            return_state=True,
            time_major=False)
        backpointers, last_score = crf_fwd_layer(
            inputs, initial_state, mask=mask)
        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1)

        crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)
        crf_bwd_layer = tf.keras.layers.RNN(
            crf_bwd_cell,
            return_sequences=True,
            return_state=True,
            time_major=False)
        decode_tags, _ = crf_bwd_layer(backpointers, initial_state)

        decode_tags = tf.squeeze(decode_tags, axis=[2])  # [B, T - 1]
        decode_tags = tf.concat(
            [initial_state, decode_tags],  # [B, T]
            axis=1)
        decode_tags = tf.reverse_sequence(  # [B, T]
            decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)  # [B]
        return decode_tags, best_score

    if potentials.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()
