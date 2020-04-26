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
import sys

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF, CRFLossLayer


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


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_keras_model_inference():
    logits, _, _, _, crf_layer = get_test_data_extended()

    input_tensor = tf.keras.layers.Input(shape=(3, 5))
    decoded_sequence, _, _, _ = crf_layer(input_tensor)
    model = tf.keras.Model(input_tensor, decoded_sequence)

    model.predict(logits)
    model(logits).numpy()


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


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_traing():
    logits, tags, _, _, crf_layer = get_test_data_extended()
    train_some_model(logits, tags)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_traing2():
    x_np, y_np = get_test_data()
    train_some_model(x_np, y_np)


def train_some_model(x_np, y_np):
    x_input = tf.keras.layers.Input(shape=x_np.shape[1:])
    y_input = tf.keras.layers.Input(shape=y_np.shape[1:])

    crf_outputs = CRF(5, name="L")(x_input)
    decoded_sequence, potentials, sequence_length, chain_kernel = crf_outputs

    crf_loss = CRFLossLayer()([potentials, y_input, sequence_length, chain_kernel])

    inference_model = tf.keras.Model(x_input, decoded_sequence)
    training_model = tf.keras.Model([x_input, y_input], crf_loss)

    training_model.compile("adam", loss="mae")
    training_model.fit((x_np, y_np), y=np.zeros((2,)))
    training_model.evaluate((x_np, y_np), y=np.zeros((2,)))

    inference_model.predict(x_np)

    return inference_model, training_model


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_mask_right_padding():
    x_np, y_np = get_test_data()
    mask = np.array([[1, 1, 1], [1, 1, 0]])

    x = tf.keras.layers.Input(shape=x_np.shape[1:])
    y = tf.keras.layers.Input(shape=y_np.shape[1:])
    crf_layer_outputs = CRF(5)(x, mask=tf.constant(mask))
    decoded_sequence, potentials, sequence_length, chain_kernel = crf_layer_outputs

    crf_loss = CRFLossLayer()([potentials, y, sequence_length, chain_kernel])

    inference_model = tf.keras.Model(x, decoded_sequence)
    training_model = tf.keras.Model([x, y], crf_loss)

    # check shape inference
    training_model.compile("adam", "mae")
    training_model.fit((x_np, y_np), np.zeros((2,)))
    inference_model.predict(x_np)


def test_mask_left_padding():
    x_np, y_np = get_test_data()
    mask = np.array([[0, 1, 1], [1, 1, 1]])

    x = tf.keras.layers.Input(shape=x_np.shape[1:])
    y = tf.keras.layers.Input(shape=y_np.shape[1:])
    crf_layer_outputs = CRF(5)(x, mask=tf.constant(mask))
    decoded_sequence, potentials, sequence_length, chain_kernel = crf_layer_outputs

    crf_loss = CRFLossLayer()([potentials, y, sequence_length, chain_kernel])

    training_model = tf.keras.Model([x, y], crf_loss)

    # we can only check the value of the mask
    # if we run eagerly. It's kind of a debug mode
    # otherwise we're wasting computation.
    training_model.compile("adam", "mae", run_eagerly=True)

    with pytest.raises(NotImplementedError) as context:
        training_model((x_np, y_np)).numpy()

    assert "CRF layer do not support left padding" in str(context.value)


def clone(model: tf.keras.Model, save_format: str):

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "my_model.tf")
        model.save(file_path, save_format=save_format)
        return tf.keras.models.load_model(file_path)


def assert_all_equal(array_list1, array_list2):
    for arr1, arr2 in zip(array_list1, array_list2):
        np.testing.assert_equal(arr1, arr2)


@pytest.mark.parametrize("save_format", ["h5", "tf"])
def test_serialization(save_format):

    x_np, y_np = get_test_data()
    inference_model, training_model = train_some_model(x_np, y_np)

    assert inference_model.get_layer("L") == training_model.get_layer("L")

    new_inference_model = clone(inference_model, save_format)
    np.testing.assert_equal(
        inference_model(x_np).numpy(), new_inference_model(x_np).numpy()
    )
    assert_all_equal(
        inference_model.get_layer("L").get_weights(),
        new_inference_model.get_layer("L").get_weights(),
    )

    new_training_model = clone(training_model, save_format)

    if save_format == "tf":
        pytest.skip("TODO: fixme. Some strange bug with TF model saving.")

    np.testing.assert_equal(
        training_model([x_np, y_np]).numpy(), new_training_model([x_np, y_np]).numpy()
    )
    assert_all_equal(
        training_model.get_layer("L").get_weights(),
        new_training_model.get_layer("L").get_weights(),
    )


def test_numerical_accuracy():
    logits, tags, transitions, boundary_values, crf_layer = get_test_data_extended()

    x_input = tf.keras.layers.Input(shape=logits.shape[1:])
    y_input = tf.keras.layers.Input(shape=tags.shape[1:])

    crf_outputs = crf_layer(x_input)
    decoded_sequence, potentials, sequence_length, chain_kernel = crf_outputs

    crf_loss = CRFLossLayer()([potentials, y_input, sequence_length, chain_kernel])
    training_model = tf.keras.Model([x_input, y_input], crf_loss)

    training_model.compile(loss="mae")
    log_likelihood = training_model.train_on_batch((logits, tags), np.zeros((2,)))

    # The manually computed log likelihood should
    # equal the result of crf.forward.
    expected_log_likelihood = compute_log_likelihood(
        logits, tags, transitions, boundary_values
    )
    unbatched_log_likelihood = -2 * log_likelihood

    np.testing.assert_allclose(
        expected_log_likelihood, unbatched_log_likelihood, rtol=2e-7
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
