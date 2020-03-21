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
"""Tests for Conditional Random Field loss."""

import itertools
import math
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.losses import crf
from tensorflow_addons.utils import test_utils

CRF_LOSS_OBJ_LIST = [crf.crf_loss, crf.ConditionalRandomFieldLoss()]


def get_test_data():
    logits = np.array(
        [
            [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
            [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
        ]
    )
    tags = np.array([[2, 3, 4], [3, 2, 2]])

    transitions = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    boundary_values = np.ones((5,))
    crf_layer = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=tf.keras.initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=tf.keras.initializers.Constant(boundary_values),
        name="crf_layer",
    )
    return logits, tags, transitions, boundary_values, crf_layer


@test_utils.run_all_in_graph_and_eager_modes
class ConditionalRandomFieldLossTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()

        self.logits = np.array(
            [
                [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
                [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
            ]
        )
        self.tags = np.array([[2, 3, 4], [3, 2, 2]])

        self.transitions = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        self.boundary_values = np.ones((5,))

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=tf.keras.initializers.Constant(self.transitions),
            use_boundary=True,
            boundary_initializer=tf.keras.initializers.Constant(self.boundary_values),
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
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        return manual_log_likelihood

    def _test_loss_function(self, loss_obj):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile("adam", loss=loss_obj, metrics=[tf.keras.metrics.Accuracy()])

        log_likelihood, _ = model.train_on_batch(self.logits, self.tags)

        # The manually computed log likelihood should
        # equal the result of crf.forward.
        expected_log_likelihood = self.compute_log_likelihood()
        unbatched_log_likelihood = -2 * log_likelihood

        self.assertAllClose(expected_log_likelihood, unbatched_log_likelihood)

    def test_class_loss_function(self):
        self._test_loss_function(crf.ConditionalRandomFieldLoss())

    def test_func_loss_function(self):
        self._test_loss_function(crf.crf_loss)


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_model_fit(loss_obj):
    logits, tags, _, _, crf_layer = get_test_data()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(3, 5)))
    model.add(crf_layer)
    model.compile("adam", loss=loss_obj, metrics=[tf.keras.metrics.Accuracy()])

    model.fit(logits, tags, epochs=10, batch_size=1)


def _test_dump_and_load(loss_obj, tmp_path):
    logits, tags, _, _, crf_layer = get_test_data()
    MODEL_PERSISTENCE_PATH = os.path.join(tmp_path, "test_saving_crf_model.h5")

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(3, 5)))
    model.add(crf_layer)
    model.compile("adam", loss=loss_obj, metrics=[tf.keras.metrics.Accuracy()])

    model.fit(logits, tags, epochs=10, batch_size=1)

    model.save(MODEL_PERSISTENCE_PATH)

    # no news is good news
    new_model = tf.keras.models.load_model(MODEL_PERSISTENCE_PATH)
    new_model.fit(logits, tags, epochs=10, batch_size=1)


@pytest.mark.skip("require tensorflow/tensorflow#37018 merged")
def test_dump_and_load_with_class_loss(tmp_path):
    # TODO(howl-anderson): wait for the PR merged

    _test_dump_and_load(crf.ConditionalRandomFieldLoss(), tmp_path)


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_mask_left_padding(loss_obj):

    train_x = np.array(
        [
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    train_y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X

    mask = np.array([[0, 1, 1], [1, 1, 1]])

    layer = CRF(5)

    x = tf.keras.layers.Input(shape=(3, 5))
    y = layer(x, mask=tf.constant(mask))

    # check shape inference
    model = tf.keras.models.Model(x, y)
    model.compile("adam", loss_obj)

    with pytest.raises(tf.errors.InvalidArgumentError) as context:
        model.fit(train_x, train_y)

    assert "CRF layer do not support left padding" in str(context.value)


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_mask_right_padding(loss_obj):
    train_x = np.array(
        [
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            [
                # O   B-X  I-X  B-Y  I-Y
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    train_y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X

    mask = np.array([[1, 1, 1], [1, 1, 0]])

    layer = CRF(5)

    x = tf.keras.layers.Input(shape=(3, 5))
    y = layer(x, mask=tf.constant(mask))

    # check shape inference
    model = tf.keras.models.Model(x, y)
    model.compile("adam", loss_obj)
    model.fit(train_x, train_y)


@pytest.mark.parametrize("loss_obj", CRF_LOSS_OBJ_LIST)
def test_serialization(loss_obj):
    ref_fn = loss_obj
    config = tf.keras.losses.serialize(ref_fn)
    fn = tf.keras.losses.deserialize(config)
    assert ref_fn.get_config() == fn.get_config()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
