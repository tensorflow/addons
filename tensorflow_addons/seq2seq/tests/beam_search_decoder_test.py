# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tfa.seq2seq.seq2seq.beam_search_decoder."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.seq2seq import attention_wrapper
from tensorflow_addons.seq2seq import beam_search_decoder, gather_tree


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree():
    # (max_time = 3, batch_size = 2, beam_width = 3)

    # create (batch_size, max_time, beam_width) matrix and transpose it
    predicted_ids = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]],
        dtype=np.int32,
    ).transpose([1, 0, 2])
    parent_ids = np.array(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2]], [[0, 0, 0], [1, 2, 0], [2, 1, 1]]],
        dtype=np.int32,
    ).transpose([1, 0, 2])

    # sequence_lengths is shaped (batch_size = 3)
    max_sequence_lengths = [3, 3]

    expected_result = np.array(
        [[[2, 2, 2], [6, 5, 6], [7, 8, 9]], [[2, 4, 4], [7, 6, 6], [8, 9, 10]]]
    ).transpose([1, 0, 2])

    res = gather_tree(
        predicted_ids,
        parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=11,
    )

    np.testing.assert_equal(expected_result, res)


def _test_gather_tree_from_array(depth_ndims=0, merged_batch_beam=False):
    array = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
            [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 0]],
        ]
    ).transpose([1, 0, 2])
    parent_ids = np.array(
        [
            [[0, 0, 0], [0, 1, 1], [2, 1, 2], [-1, -1, -1]],
            [[0, 0, 0], [1, 1, 0], [2, 0, 1], [0, 1, 0]],
        ]
    ).transpose([1, 0, 2])
    expected_array = np.array(
        [
            [[2, 2, 2], [6, 5, 6], [7, 8, 9], [0, 0, 0]],
            [[2, 3, 2], [7, 5, 7], [8, 9, 8], [11, 12, 0]],
        ]
    ).transpose([1, 0, 2])
    sequence_length = [[3, 3, 3], [4, 4, 3]]

    array = tf.convert_to_tensor(array, dtype=tf.float32)
    parent_ids = tf.convert_to_tensor(parent_ids, dtype=tf.int32)
    expected_array = tf.convert_to_tensor(expected_array, dtype=tf.float32)

    max_time = tf.shape(array)[0]
    batch_size = tf.shape(array)[1]
    beam_width = tf.shape(array)[2]

    def _tile_in_depth(tensor):
        # Generate higher rank tensors by concatenating tensor and
        # tensor + 1.
        for _ in range(depth_ndims):
            tensor = tf.stack([tensor, tensor + 1], -1)
        return tensor

    if merged_batch_beam:
        array = tf.reshape(array, [max_time, batch_size * beam_width])
        expected_array = tf.reshape(expected_array, [max_time, batch_size * beam_width])

    if depth_ndims > 0:
        array = _tile_in_depth(array)
        expected_array = _tile_in_depth(expected_array)

    sorted_array = beam_search_decoder.gather_tree_from_array(
        array, parent_ids, sequence_length
    )

    np.testing.assert_equal(expected_array.numpy(), sorted_array.numpy())


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_from_array_scalar():
    _test_gather_tree_from_array()


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_from_array_1d():
    _test_gather_tree_from_array(depth_ndims=1)


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_from_array_1d_with_merged_batch_beam():
    _test_gather_tree_from_array(depth_ndims=1, merged_batch_beam=True)


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_from_array_2d():
    _test_gather_tree_from_array(depth_ndims=2)


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_from_array_complex_trajectory():
    # Max. time = 7, batch = 1, beam = 5.
    array = np.expand_dims(
        np.array(
            [
                [[25, 12, 114, 89, 97]],
                [[9, 91, 64, 11, 162]],
                [[34, 34, 34, 34, 34]],
                [[2, 4, 2, 2, 4]],
                [[2, 3, 6, 2, 2]],
                [[2, 2, 2, 3, 2]],
                [[2, 2, 2, 2, 2]],
            ]
        ),
        -1,
    )
    parent_ids = np.array(
        [
            [[0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0]],
            [[0, 1, 2, 3, 4]],
            [[0, 0, 1, 2, 1]],
            [[0, 1, 1, 2, 3]],
            [[0, 1, 3, 1, 2]],
            [[0, 1, 2, 3, 4]],
        ]
    )
    expected_array = np.expand_dims(
        np.array(
            [
                [[25, 25, 25, 25, 25]],
                [[9, 9, 91, 9, 9]],
                [[34, 34, 34, 34, 34]],
                [[2, 4, 2, 4, 4]],
                [[2, 3, 6, 3, 6]],
                [[2, 2, 2, 3, 2]],
                [[2, 2, 2, 2, 2]],
            ]
        ),
        -1,
    )
    sequence_length = [[4, 6, 4, 7, 6]]

    array = tf.convert_to_tensor(array, dtype=tf.float32)
    parent_ids = tf.convert_to_tensor(parent_ids, dtype=tf.int32)
    expected_array = tf.convert_to_tensor(expected_array, dtype=tf.float32)

    sorted_array = beam_search_decoder.gather_tree_from_array(
        array, parent_ids, sequence_length
    )

    np.testing.assert_equal(expected_array.numpy(), sorted_array.numpy())


def basic_test_array_shape_dynamic_checks(
    static_shape, dynamic_shape, batch_size, beam_width, is_valid=True
):
    @tf.function(input_signature=(tf.TensorSpec(dynamic_shape, dtype=tf.float32),))
    def _test_body(t):
        beam_search_decoder._check_batch_beam(t, batch_size, beam_width)

    t = tf.random.uniform(static_shape, dtype=tf.float32)
    if is_valid:
        _test_body(t)
    else:
        with pytest.raises(tf.errors.InvalidArgumentError):
            _test_body(t)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_array_shape_dynamic_checks():
    basic_test_array_shape_dynamic_checks(
        (8, 4, 5, 10), (None, None, 5, 10), 4, 5, is_valid=True
    )
    basic_test_array_shape_dynamic_checks(
        (8, 20, 10), (None, None, 10), 4, 5, is_valid=True
    )
    basic_test_array_shape_dynamic_checks(
        (8, 21, 10), (None, None, 10), 4, 5, is_valid=False
    )
    basic_test_array_shape_dynamic_checks(
        (8, 4, 6, 10), (None, None, None, 10), 4, 5, is_valid=False
    )
    basic_test_array_shape_dynamic_checks((8, 4), (None, None), 4, 5, is_valid=False)


def test_array_shape_static_checks():
    assert (
        beam_search_decoder._check_static_batch_beam_maybe(
            tf.TensorShape([None, None, None]), 3, 5
        )
        is True
    )

    assert (
        beam_search_decoder._check_static_batch_beam_maybe(
            tf.TensorShape([15, None, None]), 3, 5
        )
        is True
    )
    assert (
        beam_search_decoder._check_static_batch_beam_maybe(
            tf.TensorShape([16, None, None]), 3, 5
        )
        is False
    )
    assert (
        beam_search_decoder._check_static_batch_beam_maybe(
            tf.TensorShape([3, 5, None]), 3, 5
        )
        is True
    )
    assert (
        beam_search_decoder._check_static_batch_beam_maybe(
            tf.TensorShape([3, 6, None]), 3, 5
        )
        is False
    )
    assert (
        beam_search_decoder._check_static_batch_beam_maybe(
            tf.TensorShape([5, 3, None]), 3, 5
        )
        is False
    )


def test_eos_masking():
    probs = tf.constant(
        [
            [
                [-0.2, -0.2, -0.2, -0.2, -0.2],
                [-0.3, -0.3, -0.3, 3, 0],
                [5, 6, 0, 0, 0],
            ],
            [[-0.2, -0.2, -0.2, -0.2, 0], [-0.3, -0.3, -0.1, 3, 0], [5, 6, 3, 0, 0]],
        ]
    )

    eos_token = 0
    previously_finished = np.array([[0, 1, 0], [0, 1, 1]], dtype=bool)
    masked = beam_search_decoder._mask_probs(probs, eos_token, previously_finished)
    masked = masked.numpy()

    np.testing.assert_equal(probs[0][0], masked[0][0])
    np.testing.assert_equal(probs[0][2], masked[0][2])
    np.testing.assert_equal(probs[1][0], masked[1][0])

    np.testing.assert_equal(masked[0][1][0], 0)
    np.testing.assert_equal(masked[1][1][0], 0)
    np.testing.assert_equal(masked[1][2][0], 0)

    for i in range(1, 5):
        np.testing.assert_allclose(masked[0][1][i], np.finfo("float32").min)
        np.testing.assert_allclose(masked[1][1][i], np.finfo("float32").min)
        np.testing.assert_allclose(masked[1][2][i], np.finfo("float32").min)


def test_missing_embedding_fn():
    batch_size = 6
    beam_width = 4
    cell = tf.keras.layers.LSTMCell(5)
    decoder = beam_search_decoder.BeamSearchDecoder(cell, beam_width=beam_width)
    initial_state = cell.get_initial_state(
        batch_size=batch_size * beam_width, dtype=tf.float32
    )
    start_tokens = tf.ones([batch_size], dtype=tf.int32)
    end_token = tf.constant(2, dtype=tf.int32)
    with pytest.raises(ValueError):
        decoder(None, start_tokens, end_token, initial_state)


def test_beam_step():
    batch_size = 2
    beam_width = 3
    vocab_size = 5
    end_token = 0
    length_penalty_weight = 0.6
    coverage_penalty_weight = 0.0
    output_all_scores = False

    dummy_cell_state = tf.zeros([batch_size, beam_width])
    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=tf.nn.log_softmax(tf.ones([batch_size, beam_width])),
        lengths=tf.constant(2, shape=[batch_size, beam_width], dtype=tf.int64),
        finished=tf.zeros([batch_size, beam_width], dtype=tf.bool),
        accumulated_attention_probs=(),
    )

    logits_ = np.full([batch_size, beam_width, vocab_size], 0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 2.7
    logits_[1, 2, 2] = 10.0
    logits_[1, 2, 3] = 0.2
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=tf.convert_to_tensor(batch_size),
        beam_width=beam_width,
        end_token=end_token,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
        output_all_scores=output_all_scores,
    )

    outputs_, next_state_, state_, log_probs_ = [
        outputs,
        next_beam_state,
        beam_state,
        log_probs,
    ]

    np.testing.assert_equal(
        outputs_.predicted_ids.numpy(), np.asanyarray([[3, 3, 2], [2, 2, 1]])
    )
    np.testing.assert_equal(
        outputs_.parent_ids.numpy(), np.asanyarray([[1, 0, 0], [2, 1, 0]])
    )
    np.testing.assert_equal(
        next_state_.lengths.numpy(), np.asanyarray([[3, 3, 3], [3, 3, 3]])
    )
    np.testing.assert_equal(
        next_state_.finished.numpy(),
        np.asanyarray([[False, False, False], [False, False, False]]),
    )

    expected_log_probs = []
    expected_log_probs.append(state_.log_probs[0].numpy())
    expected_log_probs.append(state_.log_probs[1].numpy())
    expected_log_probs[0][0] += log_probs_[0, 1, 3]
    expected_log_probs[0][1] += log_probs_[0, 0, 3]
    expected_log_probs[0][2] += log_probs_[0, 0, 2]
    expected_log_probs[1][0] += log_probs_[1, 2, 2]
    expected_log_probs[1][1] += log_probs_[1, 1, 2]
    expected_log_probs[1][2] += log_probs_[1, 0, 1]
    np.testing.assert_equal(
        next_state_.log_probs.numpy(), np.asanyarray(expected_log_probs)
    )


def test_step_with_eos():
    batch_size = 2
    beam_width = 3
    vocab_size = 5
    end_token = 0
    length_penalty_weight = 0.6
    coverage_penalty_weight = 0.0
    output_all_scores = False

    dummy_cell_state = tf.zeros([batch_size, beam_width])
    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=tf.nn.log_softmax(tf.ones([batch_size, beam_width])),
        lengths=tf.convert_to_tensor([[2, 1, 2], [2, 2, 1]], dtype=tf.int64),
        finished=tf.convert_to_tensor(
            [[False, True, False], [False, False, True]], dtype=tf.bool
        ),
        accumulated_attention_probs=(),
    )

    logits_ = np.full([batch_size, beam_width, vocab_size], 0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 5.7  # why does this not work when it's 2.7?
    logits_[1, 2, 2] = 1.0
    logits_[1, 2, 3] = 0.2
    logits = tf.convert_to_tensor(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=tf.convert_to_tensor(batch_size),
        beam_width=beam_width,
        end_token=end_token,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
        output_all_scores=output_all_scores,
    )

    outputs_, next_state_, state_, log_probs_ = [
        outputs,
        next_beam_state,
        beam_state,
        log_probs,
    ]

    np.testing.assert_equal(
        outputs_.parent_ids.numpy(), np.asanyarray([[1, 0, 0], [1, 2, 0]])
    )
    np.testing.assert_equal(
        outputs_.predicted_ids.numpy(), np.asanyarray([[0, 3, 2], [2, 0, 1]])
    )
    np.testing.assert_equal(
        next_state_.lengths.numpy(), np.asanyarray([[1, 3, 3], [3, 1, 3]])
    )
    np.testing.assert_equal(
        next_state_.finished.numpy(),
        np.asanyarray([[True, False, False], [False, True, False]]),
    )

    expected_log_probs = []
    expected_log_probs.append(state_.log_probs[0].numpy())
    expected_log_probs.append(state_.log_probs[1].numpy())
    expected_log_probs[0][1] += log_probs_[0, 0, 3]
    expected_log_probs[0][2] += log_probs_[0, 0, 2]
    expected_log_probs[1][0] += log_probs_[1, 1, 2]
    expected_log_probs[1][2] += log_probs_[1, 0, 1]
    np.testing.assert_equal(
        next_state_.log_probs.numpy(), np.asanyarray(expected_log_probs)
    )


def test_large_beam_step():
    batch_size = 2
    beam_width = 8
    vocab_size = 5
    end_token = 0
    length_penalty_weight = 0.6
    coverage_penalty_weight = 0.0
    output_all_scores = False

    def get_probs():
        """this simulates the initialize method in BeamSearchDecoder."""
        log_prob_mask = tf.one_hot(
            tf.zeros([batch_size], dtype=tf.int32),
            depth=beam_width,
            on_value=True,
            off_value=False,
            dtype=tf.bool,
        )

        log_prob_zeros = tf.zeros([batch_size, beam_width], dtype=tf.float32)
        log_prob_neg_inf = tf.ones([batch_size, beam_width], dtype=tf.float32) * -np.Inf

        log_probs = tf.where(log_prob_mask, log_prob_zeros, log_prob_neg_inf)
        return log_probs

    log_probs = get_probs()
    dummy_cell_state = tf.zeros([batch_size, beam_width])

    _finished = tf.one_hot(
        tf.zeros([batch_size], dtype=tf.int32),
        depth=beam_width,
        on_value=False,
        off_value=True,
        dtype=tf.bool,
    )
    _lengths = np.zeros([batch_size, beam_width], dtype=np.int64)
    _lengths[:, 0] = 2
    _lengths = tf.constant(_lengths, dtype=tf.int64)

    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=log_probs,
        lengths=_lengths,
        finished=_finished,
        accumulated_attention_probs=(),
    )

    logits_ = np.full([batch_size, beam_width, vocab_size], 0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 2.7
    logits_[1, 2, 2] = 10.0
    logits_[1, 2, 3] = 0.2
    logits = tf.constant(logits_, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=tf.convert_to_tensor(batch_size),
        beam_width=beam_width,
        end_token=end_token,
        length_penalty_weight=length_penalty_weight,
        coverage_penalty_weight=coverage_penalty_weight,
        output_all_scores=output_all_scores,
    )

    outputs_, next_state_ = [outputs, next_beam_state]

    assert outputs_.predicted_ids[0, 0] == 3
    assert outputs_.predicted_ids[0, 1] == 2
    assert outputs_.predicted_ids[1, 0] == 1
    neg_inf = -np.Inf
    np.testing.assert_equal(
        next_state_.log_probs[:, -3:].numpy(),
        np.asanyarray([[neg_inf, neg_inf, neg_inf], [neg_inf, neg_inf, neg_inf]]),
    )
    np.testing.assert_equal(
        np.asanyarray(next_state_.log_probs[:, :-3] > neg_inf), True
    )
    np.testing.assert_equal(np.asanyarray(next_state_.lengths[:, :-3] > 0), True)
    np.testing.assert_equal(
        next_state_.lengths[:, -3:].numpy(), np.asanyarray([[0, 0, 0], [0, 0, 0]])
    )


@pytest.mark.parametrize("output_all_scores", [True, False])
@pytest.mark.parametrize("with_alignment_history", [True, False])
@pytest.mark.parametrize("has_attention", [True, False])
@pytest.mark.parametrize("time_major", [True, False])
@pytest.mark.parametrize(
    "cell_class", [tf.keras.layers.LSTMCell, tf.keras.layers.GRUCell]
)
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_beam_search_decoder(
    cell_class, time_major, has_attention, with_alignment_history, output_all_scores
):
    encoder_sequence_length = np.array([3, 2, 3, 1, 1])
    batch_size = 5
    decoder_max_time = 4
    input_depth = 7
    cell_depth = 9
    attention_depth = 6
    vocab_size = 20
    end_token = vocab_size - 1
    start_token = 0
    embedding_dim = 50
    maximum_iterations = 3
    output_layer = tf.keras.layers.Dense(vocab_size, use_bias=True, activation=None)
    beam_width = 3
    embedding = tf.random.normal([vocab_size, embedding_dim])
    cell = cell_class(cell_depth)

    if has_attention:
        attention_mechanism = attention_wrapper.BahdanauAttention(
            units=attention_depth,
        )
        cell = attention_wrapper.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_depth,
            alignment_history=with_alignment_history,
        )
        coverage_penalty_weight = 0.2
    else:
        coverage_penalty_weight = 0.0

    bsd = beam_search_decoder.BeamSearchDecoder(
        cell=cell,
        beam_width=beam_width,
        output_layer=output_layer,
        length_penalty_weight=0.0,
        coverage_penalty_weight=coverage_penalty_weight,
        output_time_major=time_major,
        maximum_iterations=maximum_iterations,
        output_all_scores=output_all_scores,
    )

    @tf.function(
        input_signature=(
            tf.TensorSpec([None, None, input_depth], dtype=tf.float32),
            tf.TensorSpec([None], dtype=tf.int32),
        )
    )
    def _beam_decode_from(memory, memory_sequence_length):
        batch_size_tensor = tf.shape(memory)[0]

        if has_attention:
            tiled_memory = beam_search_decoder.tile_batch(memory, multiplier=beam_width)
            tiled_memory_sequence_length = beam_search_decoder.tile_batch(
                memory_sequence_length, multiplier=beam_width
            )
            attention_mechanism.setup_memory(
                tiled_memory, memory_sequence_length=tiled_memory_sequence_length
            )

        cell_state = cell.get_initial_state(
            batch_size=batch_size_tensor * beam_width, dtype=tf.float32
        )

        return bsd(
            embedding,
            start_tokens=tf.fill([batch_size_tensor], start_token),
            end_token=end_token,
            initial_state=cell_state,
        )

    memory = tf.random.normal([batch_size, decoder_max_time, input_depth])
    memory_sequence_length = tf.constant(encoder_sequence_length, dtype=tf.int32)
    final_outputs, final_state, final_sequence_lengths = _beam_decode_from(
        memory, memory_sequence_length
    )

    def _t(shape):
        if time_major:
            return (shape[1], shape[0]) + shape[2:]
        return shape

    assert isinstance(final_outputs, beam_search_decoder.FinalBeamSearchDecoderOutput)
    assert isinstance(final_state, beam_search_decoder.BeamSearchDecoderState)

    beam_search_decoder_output = final_outputs.beam_search_decoder_output
    max_sequence_length = np.max(final_sequence_lengths.numpy())
    assert _t((batch_size, max_sequence_length, beam_width)) == tuple(
        final_outputs.predicted_ids.shape.as_list()
    )

    if output_all_scores:
        assert _t((batch_size, max_sequence_length, beam_width, vocab_size)) == tuple(
            beam_search_decoder_output.scores.shape.as_list()
        )

        # Check that the vocab size corresponds to the dimensions of the output.
        assert (beam_width, vocab_size) == tuple(bsd.output_size.scores.as_list())
    else:
        assert _t((batch_size, max_sequence_length, beam_width)) == tuple(
            beam_search_decoder_output.scores.shape.as_list()
        )

        # Check only the beam width corresponds to the dimensions of the output.
        assert (beam_width,) == tuple(bsd.output_size.scores.as_list())
