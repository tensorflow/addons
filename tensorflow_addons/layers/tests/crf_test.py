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


def expand_1d(data):
    """Expands 1-dimensional `Tensor`s into 2-dimensional `Tensor`s."""

    def _expand_single_1d_tensor(t):
        # Leaves `CompositeTensor`s as-is.
        if (
            isinstance(t, tf.Tensor)
            and isinstance(t.shape, tf.TensorShape)
            and t.shape.rank == 1
        ):
            return tf.expand_dims(t, axis=-1)
        return t

    return tf.nest.map_structure(_expand_single_1d_tensor, data)


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return data, None, None
    elif len(data) == 1:
        return data[0], None, None
    elif len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data[0], data[1], data[2]

    raise ValueError("Data not understood.")


class ModelWithCRFLoss(tf.keras.models.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, x, y, sample_weights, training):
        y_pred = self(x, training=training)
        _, potentials, sequence_length, chain_kernel = y_pred

        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        if sample_weights is not None:
            crf_loss = crf_loss * sample_weights

        return tf.reduce_mean(crf_loss), sum(self.losses)

    def train_step(self, data):
        data = expand_1d(data)
        x, y, sample_weight = unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"crf_loss": crf_loss, "internal_losses": internal_losses}

    def test_step(self, data):
        data = expand_1d(data)
        x, y, sample_weight = unpack_x_y_sample_weight(data)
        crf_loss, internal_losses = self.compute_loss(
            x, y, sample_weight, training=False
        )
        return {"crf_loss_val": crf_loss, "internal_losses_val": internal_losses}


def test_traing():
    x_np, y_np = get_test_data()
    train_some_model(x_np, y_np)


def train_some_model(x_np, y_np, sanity_check=True):
    x_input = tf.keras.layers.Input(shape=x_np.shape[1:])
    crf_outputs = CRF(5, name="L")(x_input)
    base_model = tf.keras.Model(x_input, crf_outputs)

    wrapper_model = ModelWithCRFLoss(base_model)

    wrapper_model.compile("adam")
    if sanity_check:
        wrapper_model.fit(x=x_np, y=y_np)
        wrapper_model.evaluate(x_np, y_np)
    wrapper_model.predict(x_np)
    return wrapper_model


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


def clone(model: tf.keras.Model):

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "my_model.tf")
        model.save(file_path)
        return tf.keras.models.load_model(file_path)


def assert_all_equal(array_list1, array_list2):
    for arr1, arr2 in zip(array_list1, array_list2):
        np.testing.assert_equal(np.array(arr1), np.array(arr2))


def test_serialization():

    x_np, y_np = get_test_data()
    model = train_some_model(x_np, y_np, sanity_check=False)

    new_model = clone(model)
    assert_all_equal(model.predict(x_np), new_model.predict(x_np))
    assert_all_equal(model.get_weights(), new_model.get_weights())


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_numerical_accuracy():
    logits, tags, transitions, boundary_values, crf_layer = get_test_data_extended()

    x_input = tf.keras.layers.Input(shape=logits.shape[1:])
    crf_outputs = crf_layer(x_input)
    base_model = tf.keras.Model(x_input, crf_outputs)
    model = ModelWithCRFLoss(base_model)

    model.compile()
    log_likelihood = model.train_on_batch(logits, tags, return_dict=True)["crf_loss"]

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
