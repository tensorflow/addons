## Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Conditional Random Field loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.losses.crf_loss import ConditionalRandomFieldLoss
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class ConditionalRandomFieldLossTest(tf.test.TestCase):
    def setUp(self):
        super(ConditionalRandomFieldLossTest, self).setUp()

        self.logits = np.array([
            [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
            [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
        ])
        self.tags = np.array([[2, 3, 4], [3, 2, 2]])

        self.transitions = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])

        self.boundary_values = np.ones((5,))

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=tf.keras.initializers.Constant(self.transitions),
            use_boundary=True,
            boundary_initializer=tf.keras.initializers.Constant(
                self.boundary_values),
            name="crf_layer",
        )

    def score(self, logits, tags):
        """Computes the likelihood score for the given sequence of tags, given
        the provided logits (and the transition weights in the CRF model)"""
        # Start with transitions from START and to END
        total = self.boundary_values[tags[0]] + self.boundary_values[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def compute_log_likelihood(self):
        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores
        # for the logits across all possible tags)
        for logits_i, tags_i in zip(self.logits, self.tags):
            numerator = self.score(logits_i, tags_i)
            all_scores = [
                self.score(logits_i, tags_j)
                for tags_j in itertools.product(range(5), repeat=3)
            ]
            denominator = math.log(
                sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        return manual_log_likelihood

    def test_loss_function_as_crf_method(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile("adam", loss={"crf_layer": self.crf.loss})
        model.summary()

        log_likelihood = model.train_on_batch(self.logits, self.tags)

        # The manually computed log likelihood should
        # equal the result of crf.forward.
        expected_log_likelihood = self.compute_log_likelihood()
        unbatched_log_likelihood = -2 * log_likelihood

        self.assertAllClose(expected_log_likelihood, unbatched_log_likelihood)

    def test_loss_function_as_layer(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        crf_loss_instance = ConditionalRandomFieldLoss(model, "crf_layer")
        model.compile("adam", loss={"crf_layer": crf_loss_instance})
        model.summary()

        log_likelihood = model.train_on_batch(self.logits, self.tags)

        # The manually computed log likelihood should
        # equal the result of crf.forward.
        expected_log_likelihood = self.compute_log_likelihood()
        unbatched_log_likelihood = -2 * log_likelihood

        self.assertAllClose(expected_log_likelihood, unbatched_log_likelihood)


if __name__ == "__main__":
    tf.test.main()
