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
"""Tests for tf.addons.seq2seq.python.loss_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow_addons.seq2seq import loss
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class LossTest(tf.test.TestCase):
    def setup(self):
        self.batch_size = 2
        self.sequence_length = 3
        self.number_of_classes = 5
        logits = [
            tf.constant(
                i + 0.5, shape=[self.batch_size, self.number_of_classes])
            for i in range(self.sequence_length)
        ]
        self.logits = tf.stack(logits, axis=1)
        targets = [
            tf.constant(i, tf.int32, shape=[self.batch_size])
            for i in range(self.sequence_length)
        ]
        self.targets = tf.stack(targets, axis=1)
        weights = [
            tf.constant(1.0, shape=[self.batch_size])
            for _ in range(self.sequence_length)
        ]
        self.weights = tf.stack(weights, axis=1)
        # expected_loss = sparse_softmax_cross_entropy_with_logits(targets,
        # logits) where targets = [0, 1, 2],
        # and logits = [[0.5] * 5, [1.5] * 5, [2.5] * 5]
        self.expected_loss = 1.60944

    def testSequenceLoss(self):
        with self.cached_session(use_gpu=True):
            self.setup()
            average_loss_per_example = loss.sequence_loss(
                self.logits,
                self.targets,
                self.weights,
                average_across_timesteps=True,
                average_across_batch=True)
            res = self.evaluate(average_loss_per_example)
            self.assertAllClose(self.expected_loss, res)

            average_loss_per_sequence = loss.sequence_loss(
                self.logits,
                self.targets,
                self.weights,
                average_across_timesteps=False,
                average_across_batch=True)
            res = self.evaluate(average_loss_per_sequence)
            compare_per_sequence = np.full((self.sequence_length),
                                           self.expected_loss)
            self.assertAllClose(compare_per_sequence, res)

            average_loss_per_batch = loss.sequence_loss(
                self.logits,
                self.targets,
                self.weights,
                average_across_timesteps=True,
                average_across_batch=False)
            res = self.evaluate(average_loss_per_batch)
            compare_per_batch = np.full((self.batch_size), self.expected_loss)
            self.assertAllClose(compare_per_batch, res)

            total_loss = loss.sequence_loss(
                self.logits,
                self.targets,
                self.weights,
                average_across_timesteps=False,
                average_across_batch=False)
            res = self.evaluate(total_loss)
            compare_total = np.full((self.batch_size, self.sequence_length),
                                    self.expected_loss)
            self.assertAllClose(compare_total, res)

    def testSequenceLossClass(self):
        with self.cached_session(use_gpu=True):
            self.setup()
            seq_loss = loss.SequenceLoss(
                average_across_timesteps=True,
                average_across_batch=True,
                sum_over_timesteps=False,
                sum_over_batch=False)
            average_loss_per_example = seq_loss(self.targets, self.logits,
                                                self.weights)
            res = self.evaluate(average_loss_per_example)
            self.assertAllClose(self.expected_loss, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=True,
                sum_over_timesteps=False,
                sum_over_batch=False)
            average_loss_per_sequence = seq_loss(self.targets, self.logits,
                                                 self.weights)
            res = self.evaluate(average_loss_per_sequence)
            compare_per_sequence = np.full((self.sequence_length),
                                           self.expected_loss)
            self.assertAllClose(compare_per_sequence, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=True,
                average_across_batch=False,
                sum_over_timesteps=False,
                sum_over_batch=False)
            average_loss_per_batch = seq_loss(self.targets, self.logits,
                                              self.weights)
            res = self.evaluate(average_loss_per_batch)
            compare_per_batch = np.full((self.batch_size), self.expected_loss)
            self.assertAllClose(compare_per_batch, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=False,
                sum_over_batch=False)
            total_loss = seq_loss(self.targets, self.logits, self.weights)
            res = self.evaluate(total_loss)
            compare_total = np.full((self.batch_size, self.sequence_length),
                                    self.expected_loss)
            self.assertAllClose(compare_total, res)

    def testSumReduction(self):
        with self.cached_session(use_gpu=True):
            self.setup()
            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=True,
                sum_over_batch=True)
            average_loss_per_example = seq_loss(self.targets, self.logits,
                                                self.weights)
            res = self.evaluate(average_loss_per_example)
            self.assertAllClose(self.expected_loss, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=False,
                sum_over_batch=True)
            average_loss_per_sequence = seq_loss(self.targets, self.logits,
                                                 self.weights)
            res = self.evaluate(average_loss_per_sequence)
            compare_per_sequence = np.full((self.sequence_length),
                                           self.expected_loss)
            self.assertAllClose(compare_per_sequence, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=True,
                sum_over_batch=False)
            average_loss_per_batch = seq_loss(self.targets, self.logits,
                                              self.weights)
            res = self.evaluate(average_loss_per_batch)
            compare_per_batch = np.full((self.batch_size), self.expected_loss)
            self.assertAllClose(compare_per_batch, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=False,
                sum_over_batch=False)
            total_loss = seq_loss(self.targets, self.logits, self.weights)
            res = self.evaluate(total_loss)
            compare_total = np.full((self.batch_size, self.sequence_length),
                                    self.expected_loss)
            self.assertAllClose(compare_total, res)

    def testWeightedSumReduction(self):
        self.setup()
        weights = [
            tf.constant(1.0, shape=[self.batch_size])
            for _ in range(self.sequence_length)
        ]
        # Make the last element in the sequence to have zero weights.
        weights[-1] = tf.constant(0.0, shape=[self.batch_size])
        self.weights = tf.stack(weights, axis=1)
        with self.cached_session(use_gpu=True):
            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=True,
                sum_over_batch=True)
            average_loss_per_example = seq_loss(self.targets, self.logits,
                                                self.weights)
            res = self.evaluate(average_loss_per_example)
            self.assertAllClose(self.expected_loss, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=False,
                sum_over_batch=True)
            average_loss_per_sequence = seq_loss(self.targets, self.logits,
                                                 self.weights)
            res = self.evaluate(average_loss_per_sequence)
            compare_per_sequence = np.full((self.sequence_length),
                                           self.expected_loss)
            # The last element in every sequence are zeros, which will be
            # filtered.
            compare_per_sequence[-1] = 0.
            self.assertAllClose(compare_per_sequence, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=True,
                sum_over_batch=False)
            average_loss_per_batch = seq_loss(self.targets, self.logits,
                                              self.weights)
            res = self.evaluate(average_loss_per_batch)
            compare_per_batch = np.full((self.batch_size), self.expected_loss)
            self.assertAllClose(compare_per_batch, res)

            seq_loss = loss.SequenceLoss(
                average_across_timesteps=False,
                average_across_batch=False,
                sum_over_timesteps=False,
                sum_over_batch=False)
            total_loss = seq_loss(self.targets, self.logits, self.weights)
            res = self.evaluate(total_loss)
            compare_total = np.full((self.batch_size, self.sequence_length),
                                    self.expected_loss)
            # The last element in every sequence are zeros, which will be
            # filtered.
            compare_total[:, -1] = 0
            self.assertAllClose(compare_total, res)

    def testZeroWeights(self):
        self.setup()
        weights = [
            tf.constant(0.0, shape=[self.batch_size])
            for _ in range(self.sequence_length)
        ]
        weights = tf.stack(weights, axis=1)
        with self.test_session(use_gpu=True):
            average_loss_per_example = loss.sequence_loss(
                self.logits,
                self.targets,
                weights,
                average_across_timesteps=True,
                average_across_batch=True)
            res = self.evaluate(average_loss_per_example)
            self.assertAllClose(0.0, res)

            average_loss_per_sequence = loss.sequence_loss(
                self.logits,
                self.targets,
                weights,
                average_across_timesteps=False,
                average_across_batch=True)
            res = self.evaluate(average_loss_per_sequence)
            compare_per_sequence = np.zeros((self.sequence_length))
            self.assertAllClose(compare_per_sequence, res)

            average_loss_per_batch = loss.sequence_loss(
                self.logits,
                self.targets,
                weights,
                average_across_timesteps=True,
                average_across_batch=False)
            res = self.evaluate(average_loss_per_batch)
            compare_per_batch = np.zeros((self.batch_size))
            self.assertAllClose(compare_per_batch, res)

            total_loss = loss.sequence_loss(
                self.logits,
                self.targets,
                weights,
                average_across_timesteps=False,
                average_across_batch=False)
            res = self.evaluate(total_loss)
            compare_total = np.zeros((self.batch_size, self.sequence_length))
            self.assertAllClose(compare_total, res)


if __name__ == '__main__':
    tf.test.main()
