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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf

from tensorflow_addons import text
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class CrfTest(tf.test.TestCase):
    def calculateSequenceScore(self, inputs, transition_params, tag_indices,
                               sequence_lengths):
        expected_unary_score = sum(
            inputs[i][tag_indices[i]] for i in range(sequence_lengths))
        expected_binary_score = sum(
            transition_params[tag_indices[i], tag_indices[i + 1]]
            for i in range(sequence_lengths - 1))
        return expected_unary_score + expected_binary_score

    def testCrfFilteredInputs(self):
        # Test both the length-1 and regular cases.
        sequence_lengths_list = [
            np.array(3, dtype=np.int32),
            np.array(1, dtype=np.int32)
        ]
        inputs_list = [
            np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                     dtype=np.float32),
            np.array([[4, 5, -3]], dtype=np.float32),
        ]
        tag_bitmap_list = [
            np.array([[True, False, False], [False, True, True], [False, True, True], [False, True, True]],
                     dtype=np.float32),
            np.array([[False, True, True]], dtype=np.float32),
        ]
        neg_inf = float("-inf")
        expected_filtered_inputs_list = [
            np.array([[4, neg_inf, neg_inf], [neg_inf, -1, 3], [neg_inf, 2, 1], [neg_inf, 0, 0]],
                     dtype=np.float32),
            np.array([[4, neg_inf, neg_inf]], dtype=np.float32),
        ]
        for sequence_lengths, inputs, tag_bitmap, expected_filtered_inputs in zip(
                sequence_lengths_list, inputs_list, tag_bitmap_list, expected_filtered_inputs_list):
            filtered_inputs = text.crf_filtered_inputs(
                inputs=tf.expand_dims(inputs, 0),
                tag_bitmap=tf.expand_dims(tag_bitmap, 0))
            filtered_inputs = tf.squeeze(filtered_inputs, [0])

            tf_filtered_inputs = self.evaluate(filtered_inputs)
            self.assertAllClose(tf_filtered_inputs, expected_filtered_inputs)

    def testCrfSequenceScore(self):
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        # Test both the length-1 and regular cases.
        sequence_lengths_list = [
            np.array(3, dtype=np.int32),
            np.array(1, dtype=np.int32)
        ]
        inputs_list = [
            np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                     dtype=np.float32),
            np.array([[4, 5, -3]], dtype=np.float32),
        ]
        tag_indices_list = [
            np.array([1, 2, 1, 0], dtype=np.int32),
            np.array([1], dtype=np.int32)
        ]
        for sequence_lengths, inputs, tag_indices in zip(
                sequence_lengths_list, inputs_list, tag_indices_list):
            sequence_score = text.crf_sequence_score(
                inputs=tf.expand_dims(inputs, 0),
                tag_indices=tf.expand_dims(tag_indices, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                transition_params=tf.constant(transition_params))
            sequence_score = tf.squeeze(sequence_score, [0])

            tf_sequence_score = self.evaluate(sequence_score)

            expected_sequence_score = self.calculateSequenceScore(
                inputs, transition_params, tag_indices, sequence_lengths)
            self.assertAllClose(tf_sequence_score, expected_sequence_score)

    def testCrfMultiTagSequenceScore(self):
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        # Test both the length-1 and regular cases.
        sequence_lengths_list = [
            np.array(3, dtype=np.int32),
            np.array(1, dtype=np.int32)
        ]
        inputs_list = [
            np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                     dtype=np.float32),
            np.array([[4, 5, -3]], dtype=np.float32),
        ]
        tag_bitmap_list = [
            np.array([[True, True, False], [True, False, True],
                      [False, True, True], [True, False, True]],
                     dtype=np.bool),
            np.array([[True, True, False]], dtype=np.bool)
        ]
        for sequence_lengths, inputs, tag_bitmap in zip(
                sequence_lengths_list, inputs_list, tag_bitmap_list):
            sequence_score = text.crf_multitag_sequence_score(
                inputs=tf.expand_dims(inputs, 0),
                tag_bitmap=tf.expand_dims(tag_bitmap, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                transition_params=tf.constant(transition_params))
            sequence_score = tf.squeeze(sequence_score, [0])
            tf_sum_sequence_score = self.evaluate(sequence_score)
            all_indices_list = [
                single_index_bitmap.nonzero()[0]
                for single_index_bitmap in tag_bitmap[:sequence_lengths]
            ]
            expected_sequence_scores = [
                self.calculateSequenceScore(inputs, transition_params, indices,
                                            sequence_lengths)
                for indices in itertools.product(*all_indices_list)
            ]
            expected_log_sum_exp_sequence_scores = np.logaddexp.reduce(
                expected_sequence_scores)
            self.assertAllClose(tf_sum_sequence_score,
                                expected_log_sum_exp_sequence_scores)

    def testCrfUnaryScore(self):
        inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                          dtype=np.float32)
        for dtype in (np.int32, np.int64):
            tag_indices = np.array([1, 2, 1, 0], dtype=dtype)
            sequence_lengths = np.array(3, dtype=np.int32)
            unary_score = text.crf_unary_score(
                tag_indices=tf.expand_dims(tag_indices, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                inputs=tf.expand_dims(inputs, 0))
            unary_score = tf.squeeze(unary_score, [0])
            tf_unary_score = self.evaluate(unary_score)
            expected_unary_score = sum(
                inputs[i][tag_indices[i]] for i in range(sequence_lengths))
            self.assertAllClose(tf_unary_score, expected_unary_score)

    def testCrfBinaryScore(self):
        tag_indices = np.array([1, 2, 1, 0], dtype=np.int32)
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        sequence_lengths = np.array(3, dtype=np.int32)
        binary_score = text.crf_binary_score(
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0),
            transition_params=tf.constant(transition_params))
        binary_score = tf.squeeze(binary_score, [0])
        tf_binary_score = self.evaluate(binary_score)
        expected_binary_score = sum(
            transition_params[tag_indices[i], tag_indices[i + 1]]
            for i in range(sequence_lengths - 1))
        self.assertAllClose(tf_binary_score, expected_binary_score)

    def testCrfLogNorm(self):
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        # Test both the length-1 and regular cases.
        sequence_lengths_list = [
            np.array(3, dtype=np.int32),
            np.array(1, dtype=np.int64)
        ]
        inputs_list = [
            np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                     dtype=np.float32),
            np.array([[3, -1, 3]], dtype=np.float32),
        ]
        tag_indices_list = [
            np.array([1, 2, 1, 0], dtype=np.int32),
            np.array([2], dtype=np.int32)
        ]

        for sequence_lengths, inputs, tag_indices in zip(
                sequence_lengths_list, inputs_list, tag_indices_list):
            num_words = inputs.shape[0]
            num_tags = inputs.shape[1]
            all_sequence_scores = []

            # Compare the dynamic program with brute force computation.
            for tag_indices in itertools.product(
                    range(num_tags), repeat=sequence_lengths):
                tag_indices = list(tag_indices)
                tag_indices.extend([0] * (num_words - sequence_lengths))
                all_sequence_scores.append(
                    text.crf_sequence_score(
                        inputs=tf.expand_dims(inputs, 0),
                        tag_indices=tf.expand_dims(tag_indices, 0),
                        sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                        transition_params=tf.constant(transition_params)))

            brute_force_log_norm = tf.reduce_logsumexp(all_sequence_scores)
            log_norm = text.crf_log_norm(
                inputs=tf.expand_dims(inputs, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                transition_params=tf.constant(transition_params))
            log_norm = tf.squeeze(log_norm, [0])
            tf_brute_force_log_norm, tf_log_norm = self.evaluate(
                [brute_force_log_norm, log_norm])

            self.assertAllClose(tf_log_norm, tf_brute_force_log_norm)

    def testCrfLogNormZeroSeqLength(self):
        """Test `crf_log_norm` when `sequence_lengths` contains one or more
        zeros."""
        inputs = tf.constant(np.ones([2, 10, 5], dtype=np.float32))
        transition_params = tf.constant(np.ones([5, 5], dtype=np.float32))
        sequence_lengths = tf.constant(np.zeros([2], dtype=np.int32))
        expected_log_norm = np.zeros([2], dtype=np.float32)
        log_norm = text.crf_log_norm(inputs, sequence_lengths,
                                     transition_params)
        tf_log_norm = self.evaluate(log_norm)
        self.assertAllClose(tf_log_norm, expected_log_norm)

    def testCrfLogLikelihood(self):
        inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                          dtype=np.float32)
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        sequence_lengths = np.array(3, dtype=np.int32)
        # TODO: https://github.com/PyCQA/pylint/issues/3139
        # pylint: disable=E1136
        num_words = inputs.shape[0]
        num_tags = inputs.shape[1]
        # pylint: enable=E1136
        all_sequence_log_likelihoods = []

        # Make sure all probabilities sum to 1.
        for tag_indices in itertools.product(
                range(num_tags), repeat=sequence_lengths):
            tag_indices = list(tag_indices)
            tag_indices.extend([0] * (num_words - sequence_lengths))
            sequence_log_likelihood, _ = text.crf_log_likelihood(
                inputs=tf.expand_dims(inputs, 0),
                tag_indices=tf.expand_dims(tag_indices, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                transition_params=tf.constant(transition_params))
            all_sequence_log_likelihoods.append(sequence_log_likelihood)
        total_log_likelihood = tf.reduce_logsumexp(
            all_sequence_log_likelihoods)
        tf_total_log_likelihood = self.evaluate(total_log_likelihood)
        self.assertAllClose(tf_total_log_likelihood, 0.0)

        # check if `transition_params = None` raises an error
        text.crf_log_likelihood(
            inputs=tf.expand_dims(inputs, 0),
            tag_indices=tf.expand_dims(tag_indices, 0),
            sequence_lengths=tf.expand_dims(sequence_lengths, 0))

    def testViterbiDecode(self):
        inputs = np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                          dtype=np.float32)
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        sequence_lengths = np.array(3, dtype=np.int32)
        # TODO: https://github.com/PyCQA/pylint/issues/3139
        # pylint: disable=E1136
        num_words = inputs.shape[0]
        num_tags = inputs.shape[1]
        # pylint: enable=E1136

        all_sequence_scores = []
        all_sequences = []

        # Compare the dynamic program with brute force computation.
        for tag_indices in itertools.product(
                range(num_tags), repeat=sequence_lengths):
            tag_indices = list(tag_indices)
            tag_indices.extend([0] * (num_words - sequence_lengths))
            all_sequences.append(tag_indices)
            sequence_score = text.crf_sequence_score(
                inputs=tf.expand_dims(inputs, 0),
                tag_indices=tf.expand_dims(tag_indices, 0),
                sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                transition_params=tf.constant(transition_params))
            sequence_score = tf.squeeze(sequence_score, [0])
            all_sequence_scores.append(sequence_score)

        tf_all_sequence_scores = self.evaluate(all_sequence_scores)

        expected_max_sequence_index = np.argmax(tf_all_sequence_scores)
        expected_max_sequence = all_sequences[expected_max_sequence_index]
        expected_max_score = tf_all_sequence_scores[
            expected_max_sequence_index]

        actual_max_sequence, actual_max_score = text.viterbi_decode(
            inputs[:sequence_lengths], transition_params)

        self.assertAllClose(actual_max_score, expected_max_score)
        self.assertEqual(actual_max_sequence,
                         expected_max_sequence[:sequence_lengths])

    def testCrfDecode(self):
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        # Test both the length-1 and regular cases.
        sequence_lengths_list = [
            np.array(3, dtype=np.int32),
            np.array(1, dtype=np.int64)
        ]
        inputs_list = [
            np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                     dtype=np.float32),
            np.array([[-1, 2, 1]], dtype=np.float32),
        ]
        tag_indices_list = [
            np.array([1, 2, 1, 0], dtype=np.int32),
            np.array([2], dtype=np.int32)
        ]

        for sequence_lengths, inputs, tag_indices in zip(
                sequence_lengths_list, inputs_list, tag_indices_list):
            num_words = inputs.shape[0]
            num_tags = inputs.shape[1]

            all_sequence_scores = []
            all_sequences = []

            # Compare the dynamic program with brute force computation.
            for tag_indices in itertools.product(
                    range(num_tags), repeat=sequence_lengths):
                tag_indices = list(tag_indices)
                tag_indices.extend([0] * (num_words - sequence_lengths))
                all_sequences.append(tag_indices)
                sequence_score = text.crf_sequence_score(
                    inputs=tf.expand_dims(inputs, 0),
                    tag_indices=tf.expand_dims(tag_indices, 0),
                    sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                    transition_params=tf.constant(transition_params))
                sequence_score = tf.squeeze(sequence_score, [0])
                all_sequence_scores.append(sequence_score)

            tf_all_sequence_scores = self.evaluate(all_sequence_scores)

            expected_max_sequence_index = np.argmax(tf_all_sequence_scores)
            expected_max_sequence = all_sequences[expected_max_sequence_index]
            expected_max_score = tf_all_sequence_scores[
                expected_max_sequence_index]

            actual_max_sequence, actual_max_score = text.crf_decode(
                tf.expand_dims(inputs, 0), tf.constant(transition_params),
                tf.expand_dims(sequence_lengths, 0))
            actual_max_sequence = tf.squeeze(actual_max_sequence, [0])
            actual_max_score = tf.squeeze(actual_max_score, [0])
            tf_actual_max_sequence, tf_actual_max_score = self.evaluate(
                [actual_max_sequence, actual_max_score])

            self.assertAllClose(tf_actual_max_score, expected_max_score)
            self.assertEqual(
                list(tf_actual_max_sequence[:sequence_lengths]),
                expected_max_sequence[:sequence_lengths])

    def testCrfConstrainedDecode(self):
        transition_params = np.array([[-3, 5, -2], [3, 4, 1], [1, 2, 1]],
                                     dtype=np.float32)
        # Test both the length-1 and regular cases.
        sequence_lengths_list = [
            np.array(3, dtype=np.int32),
            np.array(1, dtype=np.int32)
        ]
        inputs_list = [
            np.array([[4, 5, -3], [3, -1, 3], [-1, 2, 1], [0, 0, 0]],
                     dtype=np.float32),
            np.array([[4, 5, -3]], dtype=np.float32),
        ]
        tag_bitmap_list = [
            np.array([[True, False, False], [False, True, True], [False, True, True], [False, True, True]],
                     dtype=np.float32),
            np.array([[False, True, True]], dtype=np.float32),
        ]
        tag_indices_list = [
            np.array([0, 2, 1, 0], dtype=np.int32),
            np.array([2], dtype=np.int32)
        ]
        for sequence_lengths, inputs, tag_bitmap, tag_indices in zip(
                sequence_lengths_list, inputs_list, tag_bitmap_list, tag_indices_list):
            num_words = inputs.shape[0]
            num_tags = inputs.shape[1]

            all_sequence_scores = []
            all_sequences = []
            filtered_inputs = text.crf_filtered_inputs(
                inputs=tf.expand_dims(inputs, 0),
                tag_bitmap=tf.expand_dims(tag_bitmap, 0))

            # Compare the dynamic program with brute force computation.
            for tag_indices in itertools.product(
                    range(num_tags), repeat=sequence_lengths):
                tag_indices = list(tag_indices)
                tag_indices.extend([0] * (num_words - sequence_lengths))
                all_sequences.append(tag_indices)
                sequence_score = text.crf_sequence_score(
                    inputs=filtered_inputs,
                    tag_indices=tf.expand_dims(tag_indices, 0),
                    sequence_lengths=tf.expand_dims(sequence_lengths, 0),
                    transition_params=tf.constant(transition_params))
                sequence_score = tf.squeeze(sequence_score, [0])
                all_sequence_scores.append(sequence_score)

            tf_all_sequence_scores = self.evaluate(all_sequence_scores)

            expected_max_sequence_index = np.argmax(tf_all_sequence_scores)
            expected_max_sequence = all_sequences[expected_max_sequence_index]
            expected_max_score = tf_all_sequence_scores[
                expected_max_sequence_index]

            actual_max_sequence, actual_max_score = text.crf_constrained_decode(
                tf.expand_dims(inputs, 0), tf.expand_dims(tag_bitmap, 0), tf.constant(transition_params),
                tf.expand_dims(sequence_lengths, 0))
            actual_max_sequence = tf.squeeze(actual_max_sequence, [0])
            actual_max_score = tf.squeeze(actual_max_score, [0])
            tf_actual_max_sequence, tf_actual_max_score = self.evaluate(
                [actual_max_sequence, actual_max_score])

            self.assertAllClose(tf_actual_max_score, expected_max_score)
            self.assertEqual(
                list(tf_actual_max_sequence[:sequence_lengths]),
                expected_max_sequence[:sequence_lengths])

    def testCrfDecodeZeroSeqLength(self):
        """Test that crf_decode works when sequence_length contains one or more
        zeros."""
        inputs = tf.constant(np.ones([2, 10, 5], dtype=np.float32))
        transition_params = tf.constant(np.ones([5, 5], dtype=np.float32))
        sequence_lengths = tf.constant(np.zeros([2], dtype=np.int32))
        tags, scores = text.crf_decode(inputs, transition_params,
                                       sequence_lengths)
        tf_tags, tf_scores = self.evaluate([tags, scores])
        self.assertEqual(len(tf_tags.shape), 2)
        self.assertEqual(len(tf_scores.shape), 1)

    def testDifferentDtype(self):
        inputs = np.ones([16, 20, 5], dtype=np.float32)
        tags = tf.convert_to_tensor(np.ones([16, 20], dtype=np.int64))
        seq_lens = np.ones([16], dtype=np.int64) * 20

        loss, _ = text.crf_log_likelihood(
            inputs=inputs, tag_indices=tags, sequence_lengths=seq_lens)

    def testTfFunction(self):
        batch_size = 4
        num_tags = 10
        input_signature = (tf.TensorSpec([None, None, num_tags]),
                           tf.TensorSpec([num_tags, num_tags]),
                           tf.TensorSpec([None], dtype=tf.int32))
        crf_decode = tf.function(input_signature=input_signature)(
            text.crf_decode)
        potentials = tf.random.uniform([batch_size, 1, num_tags])
        transition_params = tf.random.uniform([num_tags, num_tags])
        sequence_length = tf.ones([batch_size], dtype=tf.int32)
        crf_decode(potentials, transition_params, sequence_length)


if __name__ == "__main__":
    tf.test.main()
