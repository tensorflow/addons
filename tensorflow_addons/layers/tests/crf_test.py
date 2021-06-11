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
"""Tests for Conditional Random Field layer."""

import itertools
import os
import math
import tempfile

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.text.crf import crf_log_likelihood
from tensorflow_addons.utils import test_utils


def get_test_data():
    x = np.array(
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
    y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X
    return x, y


def get_test_data_extended():
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


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_keras_model_inference():
    logits, _, _, _, crf_layer = get_test_data_extended()

    input_tensor = tf.keras.layers.Input(shape=(3, 5))
    decoded_sequence, _, _, _ = crf_layer(input_tensor)
    model = tf.keras.Model(input_tensor, decoded_sequence)

    model.predict(logits)
    model(logits).numpy()


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_unmasked_viterbi_decode():
    x_np, y_np = get_test_data()

    transitions = np.ones([5, 5])
    boundary_value = np.ones(5)

    layer = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=tf.keras.initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=tf.keras.initializers.Constant(boundary_value),
    )

    decoded_sequence, _, _, _ = layer(x_np)
    decoded_sequence = decoded_sequence.numpy()
    np.testing.assert_equal(decoded_sequence, y_np)
    assert decoded_sequence.dtype == np.int32


def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        _, potentials, sequence_length, chain_kernel = y_pred

        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        return tf.reduce_mean(crf_loss), sum(self.losses)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)

        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"crf_loss": crf_loss, "internal_losses": internal_losses}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_losses = self.compute_loss(x, y, sample_weight)
        return {"crf_loss_val": crf_loss, "internal_losses_val": internal_losses}


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_traing():
    x_np, y_np = get_test_data()
    get_some_model(x_np, y_np)


def get_some_model(x_np, y_np, sanity_check=True):
    x_input = tf.keras.layers.Input(shape=x_np.shape[1:])
    crf_outputs = CRF(5, name="L")(x_input)
    base_model = tf.keras.Model(x_input, crf_outputs)

    model = ModelWithCRFLoss(base_model)

    model.compile("adam")
    if sanity_check:
        model.fit(x=x_np, y=y_np)
        model.evaluate(x_np, y_np)
    model.predict(x_np)
    return model


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_mask_right_padding():
    x_np, y_np = get_test_data()
    mask = np.array([[1, 1, 1], [1, 1, 0]])

    x = tf.keras.layers.Input(shape=x_np.shape[1:])

    crf_layer_outputs = CRF(5)(x, mask=tf.constant(mask))

    base_model = tf.keras.Model(x, crf_layer_outputs)
    model = ModelWithCRFLoss(base_model)

    # check shape inference
    model.compile("adam")
    old_weights = model.get_weights()
    model.fit(x_np, y_np)
    new_weights = model.get_weights()

    # we check that the weights were updated during the training phase.
    with pytest.raises(AssertionError):
        assert_all_equal(old_weights, new_weights)

    model.predict(x_np)


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_mask_left_padding():
    x_np, y_np = get_test_data()
    mask = np.array([[0, 1, 1], [1, 1, 1]])

    x = tf.keras.layers.Input(shape=x_np.shape[1:])
    crf_layer_outputs = CRF(5)(x, mask=tf.constant(mask))

    base_model = tf.keras.Model(x, crf_layer_outputs)
    model = ModelWithCRFLoss(base_model)

    # we can only check the value of the mask
    # if we run eagerly. It's kind of a debug mode
    # otherwise we're wasting computation.
    model.compile("adam", run_eagerly=True)

    with pytest.raises(NotImplementedError) as context:
        model(x_np).numpy()

    assert "CRF layer do not support left padding" in str(context.value)


def clone(model: ModelWithCRFLoss, inference_only=True):

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "my_model.tf")
        model.save(file_path)
        new_model = tf.keras.models.load_model(file_path)

    if not inference_only:
        # since tf doesn't save the python code of train_step and test_step
        # we need to call the wrapper again.
        # This may change, maybe later on tf will save train_step and test_step.
        new_model_with_wrapper = ModelWithCRFLoss(new_model.base_model)

        # this works, but it may be cleaner to do a copy of the optimizer
        new_model_with_wrapper.compile(optimizer=new_model.optimizer)
        new_model = new_model_with_wrapper

    return new_model


def assert_all_equal(array_list1, array_list2):
    for arr1, arr2 in zip(array_list1, array_list2):
        np.testing.assert_equal(np.array(arr1), np.array(arr2))


@pytest.mark.parametrize("inference_only", [True, False])
def test_serialization(inference_only):

    x_np, y_np = get_test_data()
    model = get_some_model(x_np, y_np, sanity_check=False)

    new_model = clone(model, inference_only)
    if inference_only:
        assert_all_equal(model.predict(x_np), new_model.predict(x_np))
        assert_all_equal(model.get_weights(), new_model.get_weights())
    else:
        original_loss = model.train_on_batch(x_np, y_np, return_dict=True)["crf_loss"]
        clone_loss = new_model.train_on_batch(x_np, y_np, return_dict=True)["crf_loss"]
        assert_all_equal(model.get_weights(), new_model.get_weights())
        assert original_loss == clone_loss


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_numerical_accuracy():
    logits, tags, transitions, boundary_values, crf_layer = get_test_data_extended()

    x_input = tf.keras.layers.Input(shape=logits.shape[1:])
    crf_outputs = crf_layer(x_input)
    base_model = tf.keras.Model(x_input, crf_outputs)
    model = ModelWithCRFLoss(base_model)

    model.compile(optimizer="Adam")
    log_likelihood = model.train_on_batch(logits, tags, return_dict=True)["crf_loss"]

    # The manually computed log likelihood should
    # equal the result of crf.forward.
    expected_log_likelihood = compute_log_likelihood(
        logits, tags, transitions, boundary_values
    )
    unbatched_log_likelihood = -2 * log_likelihood

    test_utils.assert_allclose_according_to_type(
        expected_log_likelihood, unbatched_log_likelihood, rtol=5e-5
    )


def compute_log_likelihood(logits, tags, transitions, boundary_values):
    # Now compute the log-likelihood manually
    manual_log_likelihood = 0.0

    # For each instance, manually compute the numerator
    # (which is just the score for the logits and actual tags)
    # and the denominator
    # (which is the log-sum-exp of the scores
    # for the logits across all possible tags)
    for logits_i, tags_i in zip(logits, tags):
        numerator = score_logits(logits_i, tags_i, transitions, boundary_values)
        all_scores = [
            score_logits(logits_i, tags_j, transitions, boundary_values)
            for tags_j in itertools.product(range(5), repeat=3)
        ]
        denominator = math.log(sum(math.exp(score) for score in all_scores))
        # And include them in the manual calculation.
        manual_log_likelihood += numerator - denominator

    return manual_log_likelihood


def score_logits(logits, tags, transitions, boundary_values):
    """Computes the likelihood score for the given sequence of tags, given
    the provided logits (and the transition weights in the CRF model)"""
    # Start with transitions from START and to END
    total = boundary_values[tags[0]] + boundary_values[tags[-1]]
    # Add in all the intermediate transitions
    for tag, next_tag in zip(tags, tags[1:]):
        total += transitions[tag, next_tag]
    # Add in the logits for the observed tags
    for logit, tag in zip(logits, tags):
        total += logit[tag]
    return total
