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

import sys
import pytest
from absl.testing import parameterized
import numpy as np

import tensorflow as tf

from tensorflow_addons.seq2seq import attention_wrapper
from tensorflow_addons.seq2seq import basic_decoder
from tensorflow_addons.seq2seq import sampler as sampler_py
from tensorflow_addons.utils import test_utils


@test_utils.keras_parameterized.run_all_keras_modes
class BasicDecoderTest(test_utils.keras_parameterized.TestCase):
    """Unit test for basic_decoder.BasicDecoder."""

    @parameterized.named_parameters(
        ("use_output_layer", True), ("without_output_layer", False)
    )
    def testStepWithTrainingHelperOutputLayer(self, use_output_layer):
        sequence_length = [3, 4, 3, 1, 0]
        batch_size = 5
        max_time = 8
        input_depth = 7
        cell_depth = 10
        output_layer_depth = 3

        with self.cached_session(use_gpu=True):
            inputs = np.random.randn(batch_size, max_time, input_depth).astype(
                np.float32
            )
            input_t = tf.constant(inputs)
            cell = tf.keras.layers.LSTMCell(cell_depth)
            sampler = sampler_py.TrainingSampler(time_major=False)
            if use_output_layer:
                output_layer = tf.keras.layers.Dense(output_layer_depth, use_bias=False)
                expected_output_depth = output_layer_depth
            else:
                output_layer = None
                expected_output_depth = cell_depth
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
            my_decoder = basic_decoder.BasicDecoder(
                cell=cell, sampler=sampler, output_layer=output_layer
            )

            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                input_t, initial_state=initial_state, sequence_length=sequence_length
            )
            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(
                    expected_output_depth, tf.TensorShape([])
                ),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertLen(step_state, 2)
            self.assertIsInstance(step_outputs, basic_decoder.BasicDecoderOutput)
            self.assertEqual(
                (batch_size, expected_output_depth), step_outputs[0].get_shape()
            )
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

            if use_output_layer:
                # The output layer was accessed
                self.assertEqual(len(output_layer.variables), 1)

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            self.assertAllEqual(
                [False, False, False, False, True], eval_result["first_finished"]
            )
            self.assertAllEqual(
                [False, False, False, True, True], eval_result["step_finished"]
            )
            self.assertEqual(
                output_dtype.sample_id, eval_result["step_outputs"].sample_id.dtype
            )
            self.assertAllEqual(
                np.argmax(eval_result["step_outputs"].rnn_output, -1),
                eval_result["step_outputs"].sample_id,
            )

    @parameterized.named_parameters(
        ("sequence_length_only", False), ("mask_only", True), ("no_mask", None)
    )
    def testStepWithTrainingHelperMaskedInput(self, use_mask):
        batch_size = 5
        max_time = 8
        sequence_length = (
            [max_time] * batch_size if use_mask is None else [3, 4, 3, 1, 0]
        )
        sequence_length = np.array(sequence_length, dtype=np.int32)
        mask = [[True] * l + [False] * (max_time - l) for l in sequence_length]
        input_depth = 7
        cell_depth = 10
        output_layer_depth = 3

        with self.cached_session(use_gpu=True):
            inputs = np.random.randn(batch_size, max_time, input_depth).astype(
                np.float32
            )
            input_t = tf.constant(inputs)
            cell = tf.keras.layers.LSTMCell(cell_depth)
            sampler = sampler_py.TrainingSampler(time_major=False)
            output_layer = tf.keras.layers.Dense(output_layer_depth, use_bias=False)
            expected_output_depth = output_layer_depth
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
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
                    input_t,
                    initial_state=initial_state,
                    sequence_length=sequence_length,
                )

            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(
                    expected_output_depth, tf.TensorShape([])
                ),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertLen(step_state, 2)
            assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
            self.assertEqual(
                (batch_size, expected_output_depth), step_outputs[0].get_shape()
            )
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

            self.assertLen(output_layer.variables, 1)

            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            self.assertAllEqual(sequence_length == 0, eval_result["first_finished"])
            self.assertAllEqual(
                (np.maximum(sequence_length - 1, 0) == 0), eval_result["step_finished"]
            )
            self.assertEqual(
                output_dtype.sample_id, eval_result["step_outputs"].sample_id.dtype
            )
            self.assertAllEqual(
                np.argmax(eval_result["step_outputs"].rnn_output, -1),
                eval_result["step_outputs"].sample_id,
            )

    def testStepWithGreedyEmbeddingHelper(self):
        batch_size = 5
        vocabulary_size = 7
        cell_depth = vocabulary_size  # cell's logits must match vocabulary size
        input_depth = 10
        start_tokens = np.random.randint(0, vocabulary_size, size=batch_size)
        end_token = 1

        with self.cached_session(use_gpu=True):
            embeddings = np.random.randn(vocabulary_size, input_depth).astype(
                np.float32
            )
            embeddings_t = tf.constant(embeddings)
            cell = tf.keras.layers.LSTMCell(vocabulary_size)
            sampler = sampler_py.GreedyEmbeddingSampler()
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                embeddings_t,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=initial_state,
            )
            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertLen(step_state, 2)
            self.assertTrue(isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
            self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            expected_sample_ids = np.argmax(eval_result["step_outputs"].rnn_output, -1)
            expected_step_finished = expected_sample_ids == end_token
            expected_step_next_inputs = embeddings[expected_sample_ids]
            self.assertAllEqual(
                [False, False, False, False, False], eval_result["first_finished"]
            )
            self.assertAllEqual(expected_step_finished, eval_result["step_finished"])
            self.assertEqual(
                output_dtype.sample_id, eval_result["step_outputs"].sample_id.dtype
            )
            self.assertAllEqual(
                expected_sample_ids, eval_result["step_outputs"].sample_id
            )
            self.assertAllEqual(
                expected_step_next_inputs, eval_result["step_next_inputs"]
            )

    def testStepWithSampleEmbeddingHelper(self):
        batch_size = 5
        vocabulary_size = 7
        cell_depth = vocabulary_size  # cell's logits must match vocabulary size
        input_depth = 10
        np.random.seed(0)
        start_tokens = np.random.randint(0, vocabulary_size, size=batch_size)
        end_token = 1

        with self.cached_session(use_gpu=True):
            embeddings = np.random.randn(vocabulary_size, input_depth).astype(
                np.float32
            )
            embeddings_t = tf.constant(embeddings)
            cell = tf.keras.layers.LSTMCell(vocabulary_size)
            sampler = sampler_py.SampleEmbeddingSampler(seed=0)
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                embeddings_t,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=initial_state,
            )
            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertTrue(step_state, 2)
            self.assertTrue(isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
            self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            sample_ids = eval_result["step_outputs"].sample_id
            self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
            expected_step_finished = sample_ids == end_token
            expected_step_next_inputs = embeddings[sample_ids]
            self.assertAllEqual(expected_step_finished, eval_result["step_finished"])
            self.assertAllEqual(
                expected_step_next_inputs, eval_result["step_next_inputs"]
            )

    def testStepWithScheduledEmbeddingTrainingHelper(self):
        sequence_length = [3, 4, 3, 1, 0]
        batch_size = 5
        max_time = 8
        input_depth = 7
        vocabulary_size = 10

        with self.cached_session(use_gpu=True):
            inputs = np.random.randn(batch_size, max_time, input_depth).astype(
                np.float32
            )
            input_t = tf.constant(inputs)
            embeddings = np.random.randn(vocabulary_size, input_depth).astype(
                np.float32
            )
            half = tf.constant(0.5)
            cell = tf.keras.layers.LSTMCell(vocabulary_size)
            sampler = sampler_py.ScheduledEmbeddingTrainingSampler(
                sampling_probability=half, time_major=False
            )
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                input_t,
                sequence_length=sequence_length,
                embedding=embeddings,
                initial_state=initial_state,
            )
            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(vocabulary_size, tf.TensorShape([])),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertLen(step_state, 2)
            self.assertTrue(isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
            self.assertEqual((batch_size, vocabulary_size), step_outputs[0].get_shape())
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, vocabulary_size), first_state[0].get_shape())
            self.assertEqual((batch_size, vocabulary_size), first_state[1].get_shape())
            self.assertEqual((batch_size, vocabulary_size), step_state[0].get_shape())
            self.assertEqual((batch_size, vocabulary_size), step_state[1].get_shape())
            self.assertEqual((batch_size, input_depth), step_next_inputs.get_shape())

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            self.assertAllEqual(
                [False, False, False, False, True], eval_result["first_finished"]
            )
            self.assertAllEqual(
                [False, False, False, True, True], eval_result["step_finished"]
            )
            sample_ids = eval_result["step_outputs"].sample_id
            self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
            batch_where_not_sampling = np.where(sample_ids == -1)
            batch_where_sampling = np.where(sample_ids > -1)
            self.assertAllClose(
                eval_result["step_next_inputs"][batch_where_sampling],
                embeddings[sample_ids[batch_where_sampling]],
            )
            self.assertAllClose(
                eval_result["step_next_inputs"][batch_where_not_sampling],
                np.squeeze(inputs[batch_where_not_sampling, 1], axis=0),
            )

    def _testStepWithScheduledOutputTrainingHelper(
        self, sampling_probability, use_next_inputs_fn, use_auxiliary_inputs
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

        with self.cached_session(use_gpu=True):
            inputs = np.random.randn(batch_size, max_time, input_depth).astype(
                np.float32
            )
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
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )

            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)

            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                input_t,
                sequence_length=sequence_length,
                initial_state=initial_state,
                auxiliary_inputs=auxiliary_inputs,
            )
            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)

            if use_next_inputs_fn:
                output_after_next_inputs_fn = next_inputs_fn(step_outputs.rnn_output)

            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertLen(step_state, 2)
            self.assertTrue(isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
            self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

            self.evaluate(tf.compat.v1.global_variables_initializer())

            fetches = {
                "batch_size": batch_size_t,
                "first_finished": first_finished,
                "first_inputs": first_inputs,
                "first_state": first_state,
                "step_outputs": step_outputs,
                "step_state": step_state,
                "step_next_inputs": step_next_inputs,
                "step_finished": step_finished,
            }
            if use_next_inputs_fn:
                fetches["output_after_next_inputs_fn"] = output_after_next_inputs_fn

            eval_result = self.evaluate(fetches)

            self.assertAllEqual(
                [False, False, False, False, True], eval_result["first_finished"]
            )
            self.assertAllEqual(
                [False, False, False, True, True], eval_result["step_finished"]
            )

            sample_ids = eval_result["step_outputs"].sample_id
            self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
            batch_where_not_sampling = np.where(np.logical_not(sample_ids))
            batch_where_sampling = np.where(sample_ids)

            auxiliary_inputs_to_concat = (
                auxiliary_inputs[:, 1]
                if use_auxiliary_inputs
                else np.array([]).reshape(batch_size, 0).astype(np.float32)
            )

            expected_next_sampling_inputs = np.concatenate(
                (
                    eval_result["output_after_next_inputs_fn"][batch_where_sampling]
                    if use_next_inputs_fn
                    else eval_result["step_outputs"].rnn_output[batch_where_sampling],
                    auxiliary_inputs_to_concat[batch_where_sampling],
                ),
                axis=-1,
            )
            self.assertAllClose(
                eval_result["step_next_inputs"][batch_where_sampling],
                expected_next_sampling_inputs,
            )

            self.assertAllClose(
                eval_result["step_next_inputs"][batch_where_not_sampling],
                np.concatenate(
                    (
                        np.squeeze(inputs[batch_where_not_sampling, 1], axis=0),
                        auxiliary_inputs_to_concat[batch_where_not_sampling],
                    ),
                    axis=-1,
                ),
            )

    def testStepWithScheduledOutputTrainingHelperWithoutNextInputsFnOrAuxInput(self):
        self._testStepWithScheduledOutputTrainingHelper(
            sampling_probability=0.5,
            use_next_inputs_fn=False,
            use_auxiliary_inputs=False,
        )

    def testStepWithScheduledOutputTrainingHelperWithNextInputsFn(self):
        self._testStepWithScheduledOutputTrainingHelper(
            sampling_probability=0.5,
            use_next_inputs_fn=True,
            use_auxiliary_inputs=False,
        )

    def testStepWithScheduledOutputTrainingHelperWithAuxiliaryInputs(self):
        self._testStepWithScheduledOutputTrainingHelper(
            sampling_probability=0.5,
            use_next_inputs_fn=False,
            use_auxiliary_inputs=True,
        )

    def testStepWithScheduledOutputTrainingHelperWithNextInputsFnAndAuxInputs(self):
        self._testStepWithScheduledOutputTrainingHelper(
            sampling_probability=0.5, use_next_inputs_fn=True, use_auxiliary_inputs=True
        )

    def testStepWithScheduledOutputTrainingHelperWithNoSampling(self):
        self._testStepWithScheduledOutputTrainingHelper(
            sampling_probability=0.0, use_next_inputs_fn=True, use_auxiliary_inputs=True
        )

    def testStepWithInferenceHelperCategorical(self):
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

        with self.cached_session(use_gpu=True):
            cell = tf.keras.layers.LSTMCell(vocabulary_size)
            sampler = sampler_py.InferenceSampler(
                sample_fn,
                sample_shape=(),
                sample_dtype=tf.int32,
                end_fn=end_fn,
                next_inputs_fn=next_inputs_fn,
            )
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                start_inputs, initial_state=initial_state
            )

            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(cell_depth, tf.TensorShape([])),
                output_size,
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.int32), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            self.assertLen(first_state, 2)
            self.assertLen(step_state, 2)
            self.assertTrue(isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
            self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
            self.assertEqual((batch_size,), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            sample_ids = eval_result["step_outputs"].sample_id
            self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
            expected_step_finished = sample_ids == end_token
            expected_step_next_inputs = np.zeros((batch_size, vocabulary_size))
            expected_step_next_inputs[np.arange(batch_size), sample_ids] = 1.0
            self.assertAllEqual(expected_step_finished, eval_result["step_finished"])
            self.assertAllEqual(
                expected_step_next_inputs, eval_result["step_next_inputs"]
            )

    def testStepWithInferenceHelperMultilabel(self):
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

        with self.cached_session(use_gpu=True):
            cell = tf.keras.layers.LSTMCell(vocabulary_size)
            sampler = sampler_py.InferenceSampler(
                sample_fn,
                sample_shape=[cell_depth],
                sample_dtype=tf.bool,
                end_fn=end_fn,
                next_inputs_fn=next_inputs_fn,
            )
            initial_state = cell.get_initial_state(
                batch_size=batch_size, dtype=tf.float32
            )
            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
            (first_finished, first_inputs, first_state) = my_decoder.initialize(
                start_inputs, initial_state=initial_state
            )
            output_size = my_decoder.output_size
            output_dtype = my_decoder.output_dtype
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(cell_depth, cell_depth), output_size
            )
            self.assertEqual(
                basic_decoder.BasicDecoderOutput(tf.float32, tf.bool), output_dtype
            )

            (
                step_outputs,
                step_state,
                step_next_inputs,
                step_finished,
            ) = my_decoder.step(tf.constant(0), first_inputs, first_state)
            batch_size_t = my_decoder.batch_size

            assert len(first_state) == 2
            self.assertLen(step_state, 2)
            assert isinstance(step_outputs, basic_decoder.BasicDecoderOutput)
            self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
            self.assertEqual((batch_size, cell_depth), step_outputs[1].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
            self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
            self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
            assert (batch_size, cell_depth) == step_state[1].get_shape()

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "batch_size": batch_size_t,
                    "first_finished": first_finished,
                    "first_inputs": first_inputs,
                    "first_state": first_state,
                    "step_outputs": step_outputs,
                    "step_state": step_state,
                    "step_next_inputs": step_next_inputs,
                    "step_finished": step_finished,
                }
            )

            sample_ids = eval_result["step_outputs"].sample_id
            self.assertEqual(output_dtype.sample_id, sample_ids.dtype)
            expected_step_finished = sample_ids[:, end_token]
            expected_step_next_inputs = sample_ids.astype(np.float32)
            self.assertAllEqual(expected_step_finished, eval_result["step_finished"])
            self.assertAllEqual(
                expected_step_next_inputs, eval_result["step_next_inputs"]
            )

    def testBasicDecoderWithAttentionWrapper(self):
        units = 32
        vocab_size = 1000
        attention_mechanism = attention_wrapper.LuongAttention(units)
        cell = tf.keras.layers.LSTMCell(units)
        cell = attention_wrapper.AttentionWrapper(cell, attention_mechanism)
        output_layer = tf.keras.layers.Dense(vocab_size)
        sampler = sampler_py.TrainingSampler()
        # BasicDecoder should accept a non initialized AttentionWrapper.
        basic_decoder.BasicDecoder(cell, sampler, output_layer=output_layer)

    def testRightPaddedSequenceAssertion(self):
        right_padded_sequence = [[True, True, False, False], [True, True, True, False]]
        left_padded_sequence = [[False, False, True, True], [False, True, True, True]]

        assertion = sampler_py._check_sequence_is_right_padded(
            right_padded_sequence, False
        )
        self.evaluate(assertion)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            assertion = sampler_py._check_sequence_is_right_padded(
                left_padded_sequence, False
            )
            self.evaluate(assertion)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
