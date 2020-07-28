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
"""Tests for CRF."""

import itertools

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons import text
from tensorflow_addons.utils import test_utils


def calculate_sequence_score(inputs, transition_params, tag_indices, sequence_lengths):
    expected_unary_score = sum(
        inputs[i][tag_indices[i]] for i in range(sequence_lengths)
    )
    expected_binary_score = sum(
        transition_params[tag_indices[i], tag_indices[i + 1]]
        for i in range(sequence_lengths - 1)
    )
    return expected_unary_score + expected_binary_score


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_sequence_score(dtype):
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    # Test both the length-1 and regular cases.
    sequence_lengths_list = [
        np.array(3, dtype=np.int32),
        np.array(1, dtype=np.int32),
    ]
    inputs_list = [
        np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype),
        np.array([[4, 5, -3]], dtype=dtype),
    ]
    tag_indices_list = [
        np.array([1, 2, 1, 0], dtype=np.int32),
        np.array([1], dtype=np.int32),
    ]
    for sequence_lengths, inputs, tag_indices in zip(
        sequence_lengths_list, inputs_list, tag_indices_list
    ):
        sequence_score = text.crf_sequence_score(
            inputs=tf.expand_dims(inputs, 0),
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        sequence_score = tf.squeeze(sequence_score, [0])

        expected_sequence_score = calculate_sequence_score(
            inputs, transition_params, tag_indices, sequence_lengths
        )
        test_utils.assert_allclose_according_to_type(
            sequence_score, expected_sequence_score
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_multi_tag_sequence_score(dtype):
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    # Test both the length-1 and regular cases.
    sequence_lengths_list = [
        np.array(3, dtype=np.int32),
        np.array(1, dtype=np.int32),
    ]
    inputs_list = [
        np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype),
        np.array([[4, 5, -3]], dtype=dtype),
    ]
    tag_bitmap_list = [
        np.array(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
                [True, False, True],
            ],
            dtype=np.bool,
        ),
        np.array([[True, True, False]], dtype=np.bool),
    ]
    for sequence_lengths, inputs, tag_bitmap in zip(
        sequence_lengths_list, inputs_list, tag_bitmap_list
    ):
        sequence_score = text.crf_multitag_sequence_score(
            inputs=tf.expand_dims(inputs, 0),
            tag_bitmap=tf.expand_dims(tag_bitmap, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        sequence_score = tf.squeeze(sequence_score, [0])
        all_indices_list = [
            single_index_bitmap.nonzero()[0]
            for single_index_bitmap in tag_bitmap[:sequence_lengths]
        ]
        expected_sequence_scores = [
            calculate_sequence_score(
                inputs, transition_params, indices, sequence_lengths
            )
            for indices in itertools.product(*all_indices_list)
        ]
        expected_log_sum_exp_sequence_scores = np.logaddexp.reduce(
            expected_sequence_scores
        )
        test_utils.assert_allclose_according_to_type(
            sequence_score, expected_log_sum_exp_sequence_scores
        )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_unary_score(dtype):
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype)
    for dtype in (np.int32, np.int64):
        tag_indices = np.array([1, 2, 1, 0], dtype=dtype)
        sequence_lengths = np.array(3, dtype=np.int32)
        unary_score = text.crf_unary_score(
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            inputs=tf.expand_dims(inputs, 0),
        )
        unary_score = tf.squeeze(unary_score, [0])
        expected_unary_score = sum(
            inputs[i][tag_indices[i]] for i in range(sequence_lengths)
        )
        test_utils.assert_allclose_according_to_type(unary_score, expected_unary_score)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_binary_score(dtype):
    tag_indices = np.array([1, 2, 1, 0], dtype=np.int32)
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    sequence_lengths = np.array(3, dtype=np.int32)
    binary_score = text.crf_binary_score(
        tag_indices=tf.expand_dims(tag_indices, 0),
        sequence_lengths=tf.expand_dims(sequence_lengths, 0),
        transition_params=tf.constant(transition_params),
    )
    binary_score = tf.squeeze(binary_score, [0])
    expected_binary_score = sum(
        transition_params[tag_indices[i], tag_indices[i + 1]]
        for i in range(sequence_lengths - 1)
    )
    test_utils.assert_allclose_according_to_type(binary_score, expected_binary_score)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_log_norm(dtype):
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    # Test both the length-1 and regular cases.
    sequence_lengths_list = [
        np.array(3, dtype=np.int32),
        np.array(1, dtype=np.int64),
    ]
    inputs_list = [
        np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype),
        np.array([[3, -1, 3]], dtype=dtype),
    ]
    tag_indices_list = [
        np.array([1, 2, 1, 0], dtype=np.int32),
        np.array([2], dtype=np.int32),
    ]

    for sequence_lengths, inputs, tag_indices in zip(
        sequence_lengths_list, inputs_list, tag_indices_list
    ):
        num_words = inputs.shape[0]
        num_tags = inputs.shape[1]
        all_sequence_scores = []

        # Compare the dynamic program with brute force computation.
        for tag_indices in itertools.product(range(num_tags), repeat=sequence_lengths):
            tag_indices = list(tag_indices)
            tag_indices.extend([0] * (num_words - sequence_lengths))
            all_sequence_scores.append(
                text.crf_sequence_score(
                    inputs=tf.expand_dims(inputs, 0),
                    tag_indices=tf.expand_dims(tag_indices, 0),
                    sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                    transition_params=tf.constant(transition_params),
                )
            )

        brute_force_log_norm = tf.reduce_logsumexp(all_sequence_scores)
        log_norm = text.crf_log_norm(
            inputs=tf.expand_dims(inputs, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        log_norm = tf.squeeze(log_norm, [0])

        test_utils.assert_allclose_according_to_type(log_norm, brute_force_log_norm)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_log_norm_zero_seq_length(dtype):
    """Test `crf_log_norm` when `sequence_lengths` contains one or more
    zeros."""
    inputs = tf.constant(np.ones([2, 10, 5], dtype=dtype))
    transition_params = tf.constant(np.ones([5, 5], dtype=dtype))
    sequence_lengths = tf.constant(np.zeros([2], dtype=np.int32))
    expected_log_norm = np.zeros([2], dtype=dtype)
    log_norm = text.crf_log_norm(inputs, sequence_lengths, transition_params)
    test_utils.assert_allclose_according_to_type(log_norm, expected_log_norm)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_log_likelihood(dtype):
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype)
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    sequence_lengths = np.array(3, dtype=np.int32)

    num_words = inputs.shape[0]
    num_tags = inputs.shape[1]
    all_sequence_log_likelihoods = []

    # Make sure all probabilities sum to 1.
    for tag_indices in itertools.product(range(num_tags), repeat=sequence_lengths):
        tag_indices = list(tag_indices)
        tag_indices.extend([0] * (num_words - sequence_lengths))
        sequence_log_likelihood, _ = text.crf_log_likelihood(
            inputs=tf.expand_dims(inputs, 0),
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        all_sequence_log_likelihoods.append(sequence_log_likelihood)
    total_log_likelihood = tf.reduce_logsumexp(all_sequence_log_likelihoods)
    test_utils.assert_allclose_according_to_type(
        total_log_likelihood, 0.0, rtol=1e-6, atol=1e-6, half_rtol=2e-3, half_atol=2e-3
    )

    # check if `transition_params = None` raises an error
    text.crf_log_likelihood(
        inputs=tf.expand_dims(inputs, 0),
        tag_indices=tf.expand_dims(tag_indices, 0),
        sequence_lengths=tf.expand_dims(sequence_lengths, 0),
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_viterbi_decode(dtype):
    inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype)
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    sequence_lengths = np.array(3, dtype=np.int32)
    num_words = inputs.shape[0]
    num_tags = inputs.shape[1]

    all_sequence_scores = []
    all_sequences = []

    # Compare the dynamic program with brute force computation.
    for tag_indices in itertools.product(range(num_tags), repeat=sequence_lengths):
        tag_indices = list(tag_indices)
        tag_indices.extend([0] * (num_words - sequence_lengths))
        all_sequences.append(tag_indices)
        sequence_score = text.crf_sequence_score(
            inputs=tf.expand_dims(inputs, 0),
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params),
        )
        sequence_score = tf.squeeze(sequence_score, [0])
        all_sequence_scores.append(sequence_score)

    expected_max_sequence_index = np.argmax(all_sequence_scores)
    expected_max_sequence = all_sequences[expected_max_sequence_index]
    expected_max_score = all_sequence_scores[expected_max_sequence_index]

    actual_max_sequence, actual_max_score = text.viterbi_decode(
        inputs[:sequence_lengths], transition_params
    )

    test_utils.assert_allclose_according_to_type(actual_max_score, expected_max_score)
    assert actual_max_sequence == expected_max_sequence[:sequence_lengths]


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_crf_decode(dtype):
    transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]], dtype=dtype)
    # Test both the length-1 and regular cases.
    sequence_lengths_list = [
        np.array(3, dtype=np.int32),
        np.array(1, dtype=np.int64),
    ]
    inputs_list = [
        np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]], dtype=dtype),
        np.array([[-1, 2, 1]], dtype=dtype),
    ]
    tag_indices_list = [
        np.array([1, 2, 1, 0], dtype=np.int32),
        np.array([2], dtype=np.int32),
    ]

    for sequence_lengths, inputs, tag_indices in zip(
        sequence_lengths_list, inputs_list, tag_indices_list
    ):
        num_words = inputs.shape[0]
        num_tags = inputs.shape[1]

        all_sequence_scores = []
        all_sequences = []

        # Compare the dynamic program with brute force computation.
        for tag_indices in itertools.product(range(num_tags), repeat=sequence_lengths):
            tag_indices = list(tag_indices)
            tag_indices.extend([0] * (num_words - sequence_lengths))
            all_sequences.append(tag_indices)
            sequence_score = text.crf_sequence_score(
                inputs=tf.expand_dims(inputs, 0),
                tag_indices=tf.expand_dims(tag_indices, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                transition_params=tf.constant(transition_params),
            )
            sequence_score = tf.squeeze(sequence_score, [0])
            all_sequence_scores.append(sequence_score)

        expected_max_sequence_index = np.argmax(all_sequence_scores)
        expected_max_sequence = all_sequences[expected_max_sequence_index]
        expected_max_score = all_sequence_scores[expected_max_sequence_index]

        actual_max_sequence, actual_max_score = text.crf_decode(
            tf.expand_dims(inputs, 0),
            tf.constant(transition_params),
            tf.expand_dims(sequence_lengths, 0),
        )
        actual_max_sequence = tf.squeeze(actual_max_sequence, [0])
        actual_max_score = tf.squeeze(actual_max_score, [0])

        test_utils.assert_allclose_according_to_type(
            actual_max_score, expected_max_score, 1e-6, 1e-6
        )
        assert (
            list(actual_max_sequence[:sequence_lengths])
            == expected_max_sequence[:sequence_lengths]
        )


def test_crf_decode_zero_seq_length():
    """Test that crf_decode works when sequence_length contains one or more
    zeros."""
    inputs = tf.constant(np.ones([2, 10, 5], dtype=np.float32))
    transition_params = tf.constant(np.ones([5, 5], dtype=np.float32))
    sequence_lengths = tf.constant(np.zeros([2], dtype=np.int32))
    tags, scores = text.crf_decode(inputs, transition_params, sequence_lengths)
    assert len(tags.shape) == 2
    assert len(scores.shape) == 1


def test_different_dtype():
    inputs = np.ones([16, 20, 5], dtype=np.float32)
    tags = tf.convert_to_tensor(np.ones([16, 20], dtype=np.int64))
    seq_lens = np.ones([16], dtype=np.int64) * 20

    loss, _ = text.crf_log_likelihood(
        inputs=inputs, tag_indices=tags, sequence_lengths=seq_lens
    )


def test_tf_function():
    batch_size = 4
    num_tags = 10
    input_signature = (
        tf.TensorSpec([None, None, num_tags]),
        tf.TensorSpec([num_tags, num_tags]),
        tf.TensorSpec([None], dtype=tf.int32),
    )
    crf_decode = tf.function(input_signature=input_signature)(text.crf_decode)
    potentials = tf.random.uniform([batch_size, 1, num_tags])
    transition_params = tf.random.uniform([num_tags, num_tags])
    sequence_length = tf.ones([batch_size], dtype=tf.int32)
    crf_decode(potentials, transition_params, sequence_length)


def test_crf_decode_save_load(tmpdir):
    tf.keras.backend.clear_session()
    input_tensor = tf.keras.Input(shape=(10, 3), dtype=tf.float32, name="input_tensor")
    seq_len = tf.keras.Input(shape=(), dtype=tf.int32, name="seq_len")
    transition = tf.constant([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=tf.float32)

    output = tf.multiply(input_tensor, tf.constant(1.0))
    decoded, _ = text.crf_decode(input_tensor, transition, seq_len)

    model = tf.keras.Model(
        inputs=[input_tensor, seq_len], outputs=[output, decoded], name="example_model"
    )
    model.compile(optimizer="Adam")

    x_data = {
        "input_tensor": np.random.random_sample((5, 10, 3)).astype(dtype=np.float32),
        "seq_len": np.array([10] * 5, dtype=np.int32),
    }
    y_data = {"tf_op_layer_Mul": np.random.randint(0, 3, (5, 10))}

    model.fit(x_data, y_data)
    model.predict(
        {
            "input_tensor": tf.expand_dims(x_data["input_tensor"][0], 0),
            "seq_len": np.array([10]),
        }
    )

    temp_dir = str(tmpdir.mkdir("model"))
    tf.saved_model.save(model, temp_dir)

    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(
        temp_dir,
        custom_objects={"CrfDecodeForwardRnnCell": text.crf.CrfDecodeForwardRnnCell},
    )
    model.fit(x_data, y_data)
    model.predict(
        {
            "input_tensor": tf.expand_dims(x_data["input_tensor"][0], 0),
            "seq_len": np.array([10]),
        }
    )
