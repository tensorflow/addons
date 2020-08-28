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

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.seq2seq import loss


def get_test_data():
    batch_size = 2
    sequence_length = 3
    number_of_classes = 5
    logits = [
        tf.constant(i + 0.5, shape=[batch_size, number_of_classes])
        for i in range(sequence_length)
    ]
    logits = tf.stack(logits, axis=1)
    targets = [
        tf.constant(i, tf.int32, shape=[batch_size]) for i in range(sequence_length)
    ]
    targets = tf.stack(targets, axis=1)

    weights = [tf.constant(1.0, shape=[batch_size]) for _ in range(sequence_length)]
    weights = tf.stack(weights, axis=1)
    # expected_loss = sparse_softmax_cross_entropy_with_logits(targets,
    # logits) where targets = [0, 1, 2],
    # and logits = [[0.5] * 5, [1.5] * 5, [2.5] * 5]
    expected_loss = 1.60944
    return (
        batch_size,
        sequence_length,
        number_of_classes,
        logits,
        targets,
        weights,
        expected_loss,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("average_across_timesteps", [True, False])
@pytest.mark.parametrize("average_across_batch", [True, False])
@pytest.mark.parametrize("zero_weights", [True, False])
def test_sequence_loss(average_across_timesteps, average_across_batch, zero_weights):
    (
        batch_size,
        sequence_length,
        _,
        logits,
        targets,
        weights,
        expected_loss,
    ) = get_test_data()

    if zero_weights:
        weights = [tf.constant(0.0, shape=[batch_size]) for _ in range(sequence_length)]
        weights = tf.stack(weights, axis=1)

    computed = loss.sequence_loss(
        logits,
        targets,
        weights,
        average_across_timesteps=average_across_timesteps,
        average_across_batch=average_across_batch,
    )
    computed = computed.numpy()
    if average_across_timesteps and average_across_batch and zero_weights:
        expected = 0.0
    elif not average_across_timesteps and average_across_batch and zero_weights:
        expected = np.zeros(sequence_length)
    elif average_across_timesteps and not average_across_batch and zero_weights:
        expected = np.zeros(batch_size)
    elif not average_across_timesteps and not average_across_batch and zero_weights:
        expected = np.zeros((batch_size, sequence_length))
    elif average_across_timesteps and average_across_batch and not zero_weights:
        expected = expected_loss
    elif not average_across_timesteps and average_across_batch and not zero_weights:
        expected = np.full(sequence_length, expected_loss)
    elif average_across_timesteps and not average_across_batch and not zero_weights:
        expected = np.full(batch_size, expected_loss)
    else:
        expected = np.full((batch_size, sequence_length), expected_loss)

    np.testing.assert_allclose(computed, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("average_across_timesteps", [True, False])
@pytest.mark.parametrize("average_across_batch", [True, False])
def test_sequence_loss_class(average_across_timesteps, average_across_batch):

    (
        batch_size,
        sequence_length,
        _,
        logits,
        targets,
        weights,
        expected_loss,
    ) = get_test_data()
    seq_loss = loss.SequenceLoss(
        average_across_timesteps=average_across_timesteps,
        average_across_batch=average_across_batch,
        sum_over_timesteps=False,
        sum_over_batch=False,
    )
    average_loss_per_example = seq_loss(targets, logits, weights)
    res = average_loss_per_example.numpy()
    if average_across_timesteps and average_across_batch:
        expected = expected_loss
    elif not average_across_timesteps and average_across_batch:
        expected = np.full(sequence_length, expected_loss)
    elif average_across_timesteps and not average_across_batch:
        expected = np.full(batch_size, expected_loss)
    elif not average_across_timesteps and not average_across_batch:
        expected = np.full((batch_size, sequence_length), expected_loss)

    np.testing.assert_allclose(res, expected, atol=1e-6, rtol=1e-6)


def test_sum_reduction():
    (
        batch_size,
        sequence_length,
        _,
        logits,
        targets,
        weights,
        expected_loss,
    ) = get_test_data()
    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=True,
        sum_over_batch=True,
    )
    average_loss_per_example = seq_loss(targets, logits, weights)
    res = average_loss_per_example.numpy()
    np.testing.assert_allclose(expected_loss, res, atol=1e-6, rtol=1e-6)

    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=False,
        sum_over_batch=True,
    )
    average_loss_per_sequence = seq_loss(targets, logits, weights)
    res = average_loss_per_sequence.numpy()
    compare_per_sequence = np.full((sequence_length), expected_loss)
    np.testing.assert_allclose(compare_per_sequence, res, atol=1e-6, rtol=1e-6)

    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=True,
        sum_over_batch=False,
    )
    average_loss_per_batch = seq_loss(targets, logits, weights)
    res = average_loss_per_batch.numpy()
    compare_per_batch = np.full((batch_size), expected_loss)
    np.testing.assert_allclose(compare_per_batch, res, atol=1e-6, rtol=1e-6)

    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=False,
        sum_over_batch=False,
    )
    total_loss = seq_loss(targets, logits, weights)
    res = total_loss.numpy()
    compare_total = np.full((batch_size, sequence_length), expected_loss)
    np.testing.assert_allclose(compare_total, res, atol=1e-6, rtol=1e-6)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_weighted_sum_reduction():
    (
        batch_size,
        sequence_length,
        _,
        logits,
        targets,
        _,
        expected_loss,
    ) = get_test_data()
    weights = [tf.constant(1.0, shape=[batch_size]) for _ in range(sequence_length)]
    # Make the last element in the sequence to have zero weights.
    weights[-1] = tf.constant(0.0, shape=[batch_size])
    weights = tf.stack(weights, axis=1)
    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=True,
        sum_over_batch=True,
    )
    average_loss_per_example = seq_loss(targets, logits, weights)
    res = average_loss_per_example.numpy()
    np.testing.assert_allclose(expected_loss, res, rtol=1e-6, atol=1e-6)

    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=False,
        sum_over_batch=True,
    )
    average_loss_per_sequence = seq_loss(targets, logits, weights)
    res = average_loss_per_sequence.numpy()
    compare_per_sequence = np.full(sequence_length, expected_loss)
    # The last element in every sequence are zeros, which will be
    # filtered.
    compare_per_sequence[-1] = 0.0
    np.testing.assert_allclose(compare_per_sequence, res, rtol=1e-6, atol=1e-6)

    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=True,
        sum_over_batch=False,
    )
    average_loss_per_batch = seq_loss(targets, logits, weights)
    res = average_loss_per_batch.numpy()
    compare_per_batch = np.full(batch_size, expected_loss)
    np.testing.assert_allclose(compare_per_batch, res, rtol=1e-6, atol=1e-6)

    seq_loss = loss.SequenceLoss(
        average_across_timesteps=False,
        average_across_batch=False,
        sum_over_timesteps=False,
        sum_over_batch=False,
    )
    total_loss = seq_loss(targets, logits, weights)
    res = total_loss.numpy()
    compare_total = np.full((batch_size, sequence_length), expected_loss)
    # The last element in every sequence are zeros, which will be
    # filtered.
    compare_total[:, -1] = 0
    np.testing.assert_allclose(compare_total, res, rtol=1e-6, atol=1e-6)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_ambiguous_order():
    with pytest.raises(ValueError, match="because of ambiguous order"):
        _, _, _, logits, targets, weights, _ = get_test_data()
        seq_loss = loss.SequenceLoss(
            average_across_timesteps=False,
            average_across_batch=True,
            sum_over_timesteps=True,
            sum_over_batch=False,
        )
        seq_loss(targets, logits, weights).numpy()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_keras_compatibility():
    """To test the compatibility of SequenceLoss with Keras's built-in
    training loops, we create a fake model which always outputs a pre-
    defined set of logits.

    Then we check the calculated loss to be equal to the expected
    loss. Note that since the fake model doesn't have any trainable
    parameters, no matter how many steps we train it, it always
    outputs the same loss value.
    """
    (
        batch_size,
        sequence_length,
        number_of_classes,
        logits,
        targets,
        weights,
        expected_loss,
    ) = get_test_data()
    targets = tf.one_hot(targets, depth=number_of_classes)

    def return_logits(x):
        logits_single_row = logits[0, :, :]
        logits_batch = tf.tile(
            tf.expand_dims(logits_single_row, 0), [tf.shape(x)[0], 1, 1]
        )
        return logits_batch

    inp = tf.keras.layers.Input(shape=(sequence_length,))
    out = tf.keras.layers.Lambda(
        return_logits, output_shape=(sequence_length, number_of_classes)
    )(inp)
    model = tf.keras.models.Model(inp, out)

    loss_obj = loss.SequenceLoss()
    model.compile(optimizer="adam", loss=loss_obj, sample_weight_mode="temporal")

    # This is a fake input.
    x = tf.ones(shape=(batch_size, sequence_length))

    h = model.fit(
        x, targets, sample_weight=weights, batch_size=batch_size, steps_per_epoch=1
    )

    calculated_loss = h.history["loss"][0]
    np.testing.assert_allclose(calculated_loss, expected_loss, rtol=1e-6, atol=1e-6)
