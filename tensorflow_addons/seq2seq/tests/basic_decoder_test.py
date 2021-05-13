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
"""Tests for tfa.seq2seq.basic_decoder."""

import numpy as np
import pytest

import tensorflow as tf

from tensorflow_addons.seq2seq import attention_wrapper
from tensorflow_addons.seq2seq import basic_decoder
from tensorflow_addons.seq2seq import sampler as sampler_py


@pytest.mark.parametrize("use_output_layer", [True, False])
@pytest.mark.parametrize(
    "cell_class", [tf.keras.layers.LSTMCell, tf.keras.layers.GRUCell]
)
def test_step_with_training_helper_output_layer(cell_class, use_output_layer):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    output_layer_depth = 3

    inputs = np.random.randn(batch_size, max_time, input_depth).astype(np.float32)
    input_t = tf.constant(inputs)
    cell = cell_class(cell_depth)
    sampler = sampler_py.TrainingSampler(time_major=False)
    if use_output_layer:
        output_layer = tf.keras.layers.Dense(output_layer_depth, use_bias=False)
        expected_output_depth = output_layer_depth
    else:
        output_layer = None
        expected_output_depth = cell_depth
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(
        cell=cell, sampler=sampler, output_layer=output_layer
    )

    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        input_t, initial_state=initial_state, sequence_length=sequence_length
    )
    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(expected_output_depth, tf.TensorShape([]))
        == output_size
    )

    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    if isinstance(cell, tf.keras.layers.LSTMCell):
        assert len(first_state) == 2
        assert len(step_state) == 2
        assert (batch_size, cell_depth) == first_state[0].shape
        assert (batch_size, cell_depth) == first_state[1].shape
        assert (batch_size, cell_depth) == step_state[0].shape
        assert (batch_size, cell_depth) == step_state[1].shape
    elif isinstance(cell, tf.keras.layers.GRUCell):
        assert tf.is_tensor(first_state)
        assert tf.is_tensor(step_state)
        assert (batch_size, cell_depth) == first_state.shape
        assert (batch_size, cell_depth) == step_state.shape
    assert type(step_outputs) is basic_decoder.BasicDecoderOutput
    assert (batch_size, expected_output_depth) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape

    if use_output_layer:
        # The output layer was accessed
        assert len(output_layer.variables) == 1

    np.testing.assert_equal(
        np.asanyarray([False, False, False, False, True]), first_finished
    )
    np.testing.assert_equal(
        np.asanyarray([False, False, False, True, True]), step_finished
    )
    assert output_dtype.sample_id == step_outputs.sample_id.dtype
    np.testing.assert_equal(
        np.argmax(step_outputs.rnn_output, -1), step_outputs.sample_id
    )


@pytest.mark.parametrize("use_mask", [True, False, None])
def test_step_with_training_helper_masked_input(use_mask):
    batch_size = 5
    max_time = 8
    sequence_length = [max_time] * batch_size if use_mask is None else [3, 4, 3, 1, 0]
    sequence_length = np.array(sequence_length, dtype=np.int32)
    mask = [[True] * sl + [False] * (max_time - sl) for sl in sequence_length]
    input_depth = 7
    cell_depth = 10
    output_layer_depth = 3

    inputs = np.random.randn(batch_size, max_time, input_depth).astype(np.float32)
    input_t = tf.constant(inputs)
    cell = tf.keras.layers.LSTMCell(cell_depth)
    sampler = sampler_py.TrainingSampler(time_major=False)
    output_layer = tf.keras.layers.Dense(output_layer_depth, use_bias=False)
    expected_output_depth = output_layer_depth
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(
        cell=cell, sampler=sampler, output_layer=output_layer
    )

    if use_mask is None:
        (first_finished, first_inputs, first_state) = my_decoder.initialize(
            input_t, initial_state=initial_state
        )
    elif use_mask:
        (first_finished, first_inputs, first_state) = my_decoder.initialize(
            input_t, initial_state=initial_state, mask=mask
        )
    else:
        (first_finished, first_inputs, first_state) = my_decoder.initialize(
            input_t, initial_state=initial_state, sequence_length=sequence_length
        )

    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(expected_output_depth, tf.TensorShape([]))
        == output_size
    )

    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert type(step_outputs) is basic_decoder.BasicDecoderOutput
    assert (batch_size, expected_output_depth) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape
    assert (batch_size, cell_depth) == first_state[0].shape
    assert (batch_size, cell_depth) == first_state[1].shape
    assert (batch_size, cell_depth) == step_state[0].shape
    assert (batch_size, cell_depth) == step_state[1].shape

    assert len(output_layer.variables) == 1

    np.testing.assert_equal(sequence_length == 0, first_finished)
    np.testing.assert_equal((np.maximum(sequence_length - 1, 0) == 0), step_finished)
    assert output_dtype.sample_id == step_outputs.sample_id.dtype
    np.testing.assert_equal(
        np.argmax(step_outputs.rnn_output, -1), step_outputs.sample_id
    )


def test_step_with_greedy_embedding_helper():
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size  # cell's logits must match vocabulary size
    input_depth = 10
    start_tokens = np.random.randint(0, vocabulary_size, size=batch_size)
    end_token = 1

    embeddings = np.random.randn(vocabulary_size, input_depth).astype(np.float32)
    embeddings_t = tf.constant(embeddings)
    cell = tf.keras.layers.LSTMCell(vocabulary_size)
    sampler = sampler_py.GreedyEmbeddingSampler()
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        embeddings_t,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=initial_state,
    )
    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])) == output_size
    )
    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
    assert (batch_size, cell_depth) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape
    assert (batch_size, cell_depth) == first_state[0].shape
    assert (batch_size, cell_depth) == first_state[1].shape
    assert (batch_size, cell_depth) == step_state[0].shape
    assert (batch_size, cell_depth) == step_state[1].shape

    expected_sample_ids = np.argmax(step_outputs.rnn_output, -1)
    expected_step_finished = expected_sample_ids == end_token
    expected_step_next_inputs = embeddings[expected_sample_ids]
    np.testing.assert_equal(
        np.asanyarray([False, False, False, False, False]), first_finished
    )
    np.testing.assert_equal(expected_step_finished, step_finished)
    assert output_dtype.sample_id == step_outputs.sample_id.dtype
    np.testing.assert_equal(expected_sample_ids, step_outputs.sample_id)
    np.testing.assert_equal(expected_step_next_inputs, step_next_inputs)


def test_step_with_sample_embedding_helper():
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size  # cell's logits must match vocabulary size
    input_depth = 10
    np.random.seed(0)
    start_tokens = np.random.randint(0, vocabulary_size, size=batch_size)
    end_token = 1

    embeddings = np.random.randn(vocabulary_size, input_depth).astype(np.float32)
    embeddings_t = tf.constant(embeddings)
    cell = tf.keras.layers.LSTMCell(vocabulary_size)
    sampler = sampler_py.SampleEmbeddingSampler(seed=0)
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        embeddings_t,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=initial_state,
    )
    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])) == output_size
    )
    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
    assert (batch_size, cell_depth) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape
    assert (batch_size, cell_depth) == first_state[0].shape
    assert (batch_size, cell_depth) == first_state[1].shape
    assert (batch_size, cell_depth) == step_state[0].shape
    assert (batch_size, cell_depth) == step_state[1].shape

    sample_ids = step_outputs.sample_id.numpy()
    assert output_dtype.sample_id == sample_ids.dtype
    expected_step_finished = sample_ids == end_token
    expected_step_next_inputs = embeddings[sample_ids, :]
    np.testing.assert_equal(expected_step_finished, step_finished)
    np.testing.assert_equal(expected_step_next_inputs, step_next_inputs)


def test_step_with_scheduled_embedding_training_helper():
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    vocabulary_size = 10

    inputs = np.random.randn(batch_size, max_time, input_depth).astype(np.float32)
    input_t = tf.constant(inputs)
    embeddings = np.random.randn(vocabulary_size, input_depth).astype(np.float32)
    half = tf.constant(0.5)
    cell = tf.keras.layers.LSTMCell(vocabulary_size)
    sampler = sampler_py.ScheduledEmbeddingTrainingSampler(
        sampling_probability=half, time_major=False
    )
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        input_t,
        sequence_length=sequence_length,
        embedding=embeddings,
        initial_state=initial_state,
    )
    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(vocabulary_size, tf.TensorShape([]))
        == output_size
    )

    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
    assert (batch_size, vocabulary_size) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape
    assert (batch_size, vocabulary_size) == first_state[0].shape
    assert (batch_size, vocabulary_size) == first_state[1].shape
    assert (batch_size, vocabulary_size) == step_state[0].shape
    assert (batch_size, vocabulary_size) == step_state[1].shape
    assert (batch_size, input_depth) == step_next_inputs.shape

    np.testing.assert_equal(
        np.asanyarray([False, False, False, False, True]), first_finished
    )
    np.testing.assert_equal(
        np.asanyarray([False, False, False, True, True]), step_finished
    )
    sample_ids = step_outputs.sample_id.numpy()
    assert output_dtype.sample_id == sample_ids.dtype
    batch_where_not_sampling = np.where(sample_ids == -1)
    batch_where_sampling = np.where(sample_ids > -1)

    np.testing.assert_equal(
        step_next_inputs.numpy()[batch_where_sampling],
        embeddings[sample_ids[batch_where_sampling]],
    )
    np.testing.assert_equal(
        step_next_inputs.numpy()[batch_where_not_sampling],
        np.squeeze(inputs[batch_where_not_sampling, 1], axis=0),
    )


@pytest.mark.parametrize("use_auxiliary_inputs", [True, False])
@pytest.mark.parametrize("use_next_inputs_fn", [True, False])
@pytest.mark.parametrize("sampling_probability", [0.0, 0.5])
def test_step_with_scheduled_output_training_helper(
    sampling_probability, use_next_inputs_fn, use_auxiliary_inputs
):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = input_depth
    if use_auxiliary_inputs:
        auxiliary_input_depth = 4
        auxiliary_inputs = np.random.randn(
            batch_size, max_time, auxiliary_input_depth
        ).astype(np.float32)
    else:
        auxiliary_inputs = None

    inputs = np.random.randn(batch_size, max_time, input_depth).astype(np.float32)
    input_t = tf.constant(inputs)
    cell = tf.keras.layers.LSTMCell(cell_depth)
    sampling_probability = tf.constant(sampling_probability)

    if use_next_inputs_fn:

        def next_inputs_fn(outputs):
            # Use deterministic function for test.
            samples = tf.argmax(outputs, axis=1)
            return tf.one_hot(samples, cell_depth, dtype=tf.float32)

    else:
        next_inputs_fn = None

    sampler = sampler_py.ScheduledOutputTrainingSampler(
        sampling_probability=sampling_probability,
        time_major=False,
        next_inputs_fn=next_inputs_fn,
    )
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)

    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        input_t,
        sequence_length=sequence_length,
        initial_state=initial_state,
        auxiliary_inputs=auxiliary_inputs,
    )
    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])) == output_size
    )
    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    if use_next_inputs_fn:
        output_after_next_inputs_fn = next_inputs_fn(step_outputs.rnn_output)

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
    assert (batch_size, cell_depth) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape
    assert (batch_size, cell_depth) == first_state[0].shape
    assert (batch_size, cell_depth) == first_state[1].shape
    assert (batch_size, cell_depth) == step_state[0].shape
    assert (batch_size, cell_depth) == step_state[1].shape

    np.testing.assert_equal(
        np.asanyarray([False, False, False, False, True]), first_finished
    )
    np.testing.assert_equal(
        np.asanyarray([False, False, False, True, True]), step_finished
    )

    sample_ids = step_outputs.sample_id
    assert output_dtype.sample_id == sample_ids.dtype
    batch_where_not_sampling = np.where(np.logical_not(sample_ids))
    batch_where_sampling = np.where(sample_ids)

    auxiliary_inputs_to_concat = (
        auxiliary_inputs[:, 1]
        if use_auxiliary_inputs
        else np.array([]).reshape(batch_size, 0).astype(np.float32)
    )

    expected_next_sampling_inputs = np.concatenate(
        (
            output_after_next_inputs_fn.numpy()[batch_where_sampling]
            if use_next_inputs_fn
            else step_outputs.rnn_output.numpy()[batch_where_sampling],
            auxiliary_inputs_to_concat[batch_where_sampling],
        ),
        axis=-1,
    )

    np.testing.assert_equal(
        step_next_inputs.numpy()[batch_where_sampling], expected_next_sampling_inputs
    )

    np.testing.assert_equal(
        step_next_inputs.numpy()[batch_where_not_sampling],
        np.concatenate(
            (
                np.squeeze(inputs[batch_where_not_sampling, 1], axis=0),
                auxiliary_inputs_to_concat[batch_where_not_sampling],
            ),
            axis=-1,
        ),
    )


def test_step_with_inference_helper_categorical():
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size
    start_token = 0
    end_token = 6

    start_inputs = tf.one_hot(
        np.ones(batch_size, dtype=np.int32) * start_token, vocabulary_size
    )

    # The sample function samples categorically from the logits.
    def sample_fn(x):
        return sampler_py.categorical_sample(logits=x)

    # The next inputs are a one-hot encoding of the sampled labels.
    def next_inputs_fn(x):
        return tf.one_hot(x, vocabulary_size, dtype=tf.float32)

    def end_fn(sample_ids):
        return tf.equal(sample_ids, end_token)

    cell = tf.keras.layers.LSTMCell(vocabulary_size)
    sampler = sampler_py.InferenceSampler(
        sample_fn,
        sample_shape=(),
        sample_dtype=tf.int32,
        end_fn=end_fn,
        next_inputs_fn=next_inputs_fn,
    )
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        start_inputs, initial_state=initial_state
    )

    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert (
        basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])) == output_size
    )
    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.int32) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
    assert (batch_size, cell_depth) == step_outputs[0].shape
    assert (batch_size,) == step_outputs[1].shape
    assert (batch_size, cell_depth) == first_state[0].shape
    assert (batch_size, cell_depth) == first_state[1].shape
    assert (batch_size, cell_depth) == step_state[0].shape
    assert (batch_size, cell_depth) == step_state[1].shape

    sample_ids = step_outputs.sample_id.numpy()
    assert output_dtype.sample_id == sample_ids.dtype
    expected_step_finished = sample_ids == end_token
    expected_step_next_inputs = np.zeros((batch_size, vocabulary_size))
    expected_step_next_inputs[np.arange(batch_size), sample_ids] = 1.0
    np.testing.assert_equal(expected_step_finished, step_finished)
    np.testing.assert_equal(expected_step_next_inputs, step_next_inputs)


def test_step_with_inference_helper_multilabel():
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size
    start_token = 0
    end_token = 6

    start_inputs = tf.one_hot(
        np.ones(batch_size, dtype=np.int32) * start_token, vocabulary_size
    )

    # The sample function samples independent bernoullis from the logits.
    def sample_fn(x):
        return sampler_py.bernoulli_sample(logits=x, dtype=tf.bool)

    # The next inputs are a one-hot encoding of the sampled labels.
    def next_inputs_fn(x):
        return tf.cast(x, tf.float32)

    def end_fn(sample_ids):
        return sample_ids[:, end_token]

    cell = tf.keras.layers.LSTMCell(vocabulary_size)
    sampler = sampler_py.InferenceSampler(
        sample_fn,
        sample_shape=[cell_depth],
        sample_dtype=tf.bool,
        end_fn=end_fn,
        next_inputs_fn=next_inputs_fn,
    )
    initial_state = cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
    (first_finished, first_inputs, first_state) = my_decoder.initialize(
        start_inputs, initial_state=initial_state
    )
    output_size = my_decoder.output_size
    output_dtype = my_decoder.output_dtype
    assert basic_decoder.BasicDecoderOutput(cell_depth, cell_depth) == output_size
    assert basic_decoder.BasicDecoderOutput(tf.float32, tf.bool) == output_dtype

    (step_outputs, step_state, step_next_inputs, step_finished) = my_decoder.step(
        tf.constant(0), first_inputs, first_state
    )

    assert len(first_state) == 2
    assert len(step_state) == 2
    assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
    assert (batch_size, cell_depth) == step_outputs[0].shape
    assert (batch_size, cell_depth) == step_outputs[1].shape
    assert (batch_size, cell_depth) == first_state[0].shape
    assert (batch_size, cell_depth) == first_state[1].shape
    assert (batch_size, cell_depth) == step_state[0].shape
    assert (batch_size, cell_depth) == step_state[1].shape

    sample_ids = step_outputs.sample_id.numpy()
    assert output_dtype.sample_id == sample_ids.dtype
    expected_step_finished = sample_ids[:, end_token]
    expected_step_next_inputs = sample_ids.astype(np.float32)
    np.testing.assert_equal(expected_step_finished, step_finished)
    np.testing.assert_equal(expected_step_next_inputs, step_next_inputs)


def test_basic_decoder_with_attention_wrapper():
    units = 32
    vocab_size = 1000
    attention_mechanism = attention_wrapper.LuongAttention(units)
    cell = tf.keras.layers.LSTMCell(units)
    cell = attention_wrapper.AttentionWrapper(cell, attention_mechanism)
    output_layer = tf.keras.layers.Dense(vocab_size)
    sampler = sampler_py.TrainingSampler()
    # BasicDecoder should accept a non initialized AttentionWrapper.
    basic_decoder.BasicDecoder(cell, sampler, output_layer=output_layer)


def test_right_padded_sequence_assertion():
    right_padded_sequence = [[True, True, False, False], [True, True, True, False]]
    left_padded_sequence = [[False, False, True, True], [False, True, True, True]]

    _ = sampler_py._check_sequence_is_right_padded(right_padded_sequence, False)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = sampler_py._check_sequence_is_right_padded(left_padded_sequence, False)
