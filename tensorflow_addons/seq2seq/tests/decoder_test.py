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
"""Tests for tfa.seq2seq.decoder."""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.seq2seq import basic_decoder
from tensorflow_addons.seq2seq import sampler as sampler_py


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize(
    "maximum_iterations", [None, 1, tf.constant(1, dtype=tf.int32)]
)
@pytest.mark.parametrize("time_major", [True, False])
def test_dynamic_decode_rnn(time_major, maximum_iterations):

    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    cell = tf.keras.layers.LSTMCell(cell_depth)
    sampler = sampler_py.TrainingSampler(time_major=time_major)
    my_decoder = basic_decoder.BasicDecoder(
        cell=cell,
        sampler=sampler,
        output_time_major=time_major,
        maximum_iterations=maximum_iterations,
    )

    @tf.function(
        input_signature=(
            tf.TensorSpec([None, None, input_depth], dtype=tf.float32),
            tf.TensorSpec([None], dtype=tf.int32),
        )
    )
    def _decode(inputs, sequence_length):
        batch_size_t = tf.shape(sequence_length)[0]
        initial_state = cell.get_initial_state(
            batch_size=batch_size_t, dtype=inputs.dtype
        )
        return my_decoder(
            inputs, initial_state=initial_state, sequence_length=sequence_length
        )

    inputs = tf.random.normal([batch_size, max_time, input_depth])
    if time_major:
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
    final_outputs, _, final_sequence_length = _decode(inputs, sequence_length)

    def _t(shape):
        if time_major:
            return (shape[1], shape[0]) + shape[2:]
        return shape

    assert (batch_size,) == tuple(final_sequence_length.shape.as_list())
    # Mostly a smoke test
    time_steps = max_out
    expected_length = sequence_length
    if maximum_iterations is not None:
        time_steps = min(max_out, maximum_iterations)
        expected_length = [min(x, maximum_iterations) for x in expected_length]
    assert _t((batch_size, time_steps, cell_depth)) == final_outputs.rnn_output.shape
    assert _t((batch_size, time_steps)) == final_outputs.sample_id.shape
    np.testing.assert_array_equal(expected_length, final_sequence_length)


@pytest.mark.parametrize("use_sequence_length", [True, False])
def test_dynamic_decode_rnn_with_training_helper_matches_dynamic_rnn(
    use_sequence_length,
):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    inputs = np.random.randn(batch_size, max_time, input_depth).astype(np.float32)
    inputs = tf.constant(inputs)

    cell = tf.keras.layers.LSTMCell(cell_depth)
    zero_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    sampler = sampler_py.TrainingSampler()
    my_decoder = basic_decoder.BasicDecoder(
        cell=cell, sampler=sampler, impute_finished=use_sequence_length
    )

    (final_decoder_outputs, final_decoder_state, _,) = my_decoder(
        inputs, initial_state=zero_state, sequence_length=sequence_length
    )

    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
    mask = (
        tf.sequence_mask(sequence_length, maxlen=max_time)
        if use_sequence_length
        else None
    )
    outputs = rnn(inputs, mask=mask, initial_state=zero_state)
    final_rnn_outputs = outputs[0]
    final_rnn_state = outputs[1:]
    if use_sequence_length:
        final_rnn_outputs *= tf.cast(tf.expand_dims(mask, -1), final_rnn_outputs.dtype)

    # Decoder only runs out to max_out; ensure values are identical
    # to dynamic_rnn, which also zeros out outputs and passes along
    # state.
    np.testing.assert_allclose(
        final_decoder_outputs.rnn_output, final_rnn_outputs[:, 0:max_out, :],
    )
    if use_sequence_length:
        np.testing.assert_allclose(final_decoder_state, final_rnn_state)
