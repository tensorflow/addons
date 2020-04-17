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
"""Tests for tfa.seq2seq.attention_wrapper."""

import collections

import pytest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.seq2seq import attention_wrapper as wrapper
from tensorflow_addons.seq2seq import basic_decoder
from tensorflow_addons.seq2seq import sampler as sampler_py


class DummyData:
    def __init__(self):
        self.batch = 10
        self.timestep = 5
        self.memory_size = 6
        self.units = 8

        self.memory = np.random.randn(
            self.batch, self.timestep, self.memory_size
        ).astype(np.float32)
        self.memory_length = np.random.randint(
            low=1, high=self.timestep + 1, size=(self.batch,)
        )
        self.query = np.random.randn(self.batch, self.units).astype(np.float32)
        self.state = np.random.randn(self.batch, self.timestep).astype(np.float32)


attention_classes = [
    wrapper.LuongAttention,
    wrapper.LuongMonotonicAttention,
    wrapper.BahdanauAttention,
    wrapper.BahdanauMonotonicAttention,
]


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_attention_shape_inference(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    attention_score = attention([dummy_data.query, dummy_data.state])
    assert len(attention_score) == 2
    assert attention_score[0].shape == (dummy_data.batch, dummy_data.timestep)
    assert attention_score[1].shape == (dummy_data.batch, dummy_data.timestep)


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_get_config(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    config = attention.get_config()

    attention_from_config = attention_cls.from_config(config)
    config_from_clone = attention_from_config.get_config()

    assert config == config_from_clone


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_layer_output(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    score = attention([dummy_data.query, dummy_data.state])

    assert len(score) == 2
    assert score[0].shape == (dummy_data.batch, dummy_data.timestep)
    assert score[1].shape == (dummy_data.batch, dummy_data.timestep)


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_passing_memory_from_call(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units, dummy_data.memory)
    weights_before_query = attention.get_weights()
    ref_score = attention([dummy_data.query, dummy_data.state])

    all_weights = attention.get_weights()
    config = attention.get_config()
    # Simulate the twice invocation of calls here.
    attention_from_config = attention_cls.from_config(config)
    attention_from_config.build(dummy_data.memory.shape)
    attention_from_config.set_weights(weights_before_query)
    attention_from_config(dummy_data.memory, setup_memory=True)
    attention_from_config.build([dummy_data.query.shape, dummy_data.state.shape])
    attention_from_config.set_weights(all_weights)
    score = attention_from_config([dummy_data.query, dummy_data.state])

    np.testing.assert_allclose(ref_score, score)


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_save_load_layer(attention_cls):
    dummy_data = DummyData()
    vocab = 20
    embedding_dim = 6
    inputs = tf.keras.Input(shape=[dummy_data.timestep])
    encoder_input = tf.keras.layers.Embedding(vocab, embedding_dim, mask_zero=True)(
        inputs
    )
    encoder_output = tf.keras.layers.LSTM(
        dummy_data.memory_size, return_sequences=True
    )(encoder_input)

    attention = attention_cls(dummy_data.units, encoder_output)
    query = tf.keras.Input(shape=[dummy_data.units])
    state = tf.keras.Input(shape=[dummy_data.timestep])

    score = attention([query, state])

    x_test = np.random.randint(vocab, size=(dummy_data.batch, dummy_data.timestep))
    model = tf.keras.Model([inputs, query, state], score)
    # Fall back to v1 style Keras training loop until issue with
    # using outputs of a layer in another layer's constructor.
    model.compile("rmsprop", "mse")
    y_ref = model.predict_on_batch([x_test, dummy_data.query, dummy_data.state])

    config = model.get_config()
    weights = model.get_weights()
    loaded_model = tf.keras.Model.from_config(
        config, custom_objects={attention_cls.__name__: attention_cls}
    )
    loaded_model.set_weights(weights)

    # Fall back to v1 style Keras training loop until issue with
    # using outputs of a layer in another layer's constructor.
    loaded_model.compile("rmsprop", "mse")

    y = loaded_model.predict_on_batch([x_test, dummy_data.query, dummy_data.state])

    np.testing.assert_allclose(y_ref, y)


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_manual_memory_reset(attention_cls):
    dummy_data = DummyData()
    attention = attention_cls(dummy_data.units)

    def _compute_score(batch_size=None):
        if batch_size is None:
            batch_size = dummy_data.batch
        memory = dummy_data.memory[:batch_size]
        attention.setup_memory(
            memory, memory_sequence_length=dummy_data.memory_length[:batch_size]
        )
        assert attention.values.shape.as_list() == list(memory.shape)
        assert attention.keys.shape.as_list() == list(memory.shape)[:-1] + [
            dummy_data.units
        ]
        return attention([dummy_data.query[:batch_size], dummy_data.state[:batch_size]])

    _compute_score(batch_size=dummy_data.batch)
    variables = list(attention.variables)
    _compute_score(batch_size=dummy_data.batch - 1)

    # No new variables were created.
    for var_1, var_2 in zip(variables, list(attention.variables)):
        assert var_1 is var_2


def test_masking():
    memory = tf.ones([4, 4, 5], dtype=tf.float32)
    memory_sequence_length = tf.constant([1, 2, 3, 4], dtype=tf.int32)
    query = tf.ones([4, 5], dtype=tf.float32)
    state = None
    attention = wrapper.LuongAttention(5, memory, memory_sequence_length)
    alignment, _ = attention([query, state])
    assert np.sum(np.triu(alignment, k=1)) == 0


@pytest.mark.parametrize(
    "attention_cls", attention_classes,
)
def test_memory_re_setup(attention_cls):
    class MyModel(tf.keras.models.Model):
        def __init__(self, vocab, embedding_dim, memory_size, units):
            super().__init__()
            self.emb = tf.keras.layers.Embedding(vocab, embedding_dim, mask_zero=True)
            self.encoder = tf.keras.layers.LSTM(memory_size, return_sequences=True)
            self.attn_mch = attention_cls(units)

        def call(self, inputs):
            enc_input, query, state = inputs
            mask = self.emb.compute_mask(enc_input)
            enc_input = self.emb(enc_input)
            enc_output = self.encoder(enc_input, mask=mask)
            # To ensure manual resetting also works in the graph mode,
            # we call the attention mechanism twice.
            self.attn_mch(enc_output, mask=mask, setup_memory=True)
            self.attn_mch(enc_output, mask=mask, setup_memory=True)
            score = self.attn_mch([query, state])
            return score

    vocab = 20
    embedding_dim = 6
    num_batches = 5

    dummy_data = DummyData()
    model = MyModel(vocab, embedding_dim, dummy_data.memory_size, dummy_data.units)
    if tf.executing_eagerly():
        model.compile("rmsprop", "mse", run_eagerly=True)
    else:
        model.compile("rmsprop", "mse")

    x = np.random.randint(
        vocab, size=(num_batches * dummy_data.batch, dummy_data.timestep)
    )
    x_test = np.random.randint(
        vocab, size=(num_batches * dummy_data.batch, dummy_data.timestep)
    )
    y = np.random.randn(num_batches * dummy_data.batch, dummy_data.timestep)

    query = np.tile(dummy_data.query, [num_batches, 1])
    state = np.tile(dummy_data.state, [num_batches, 1])

    model.fit([x, query, state], (y, y), batch_size=dummy_data.batch)
    model.predict_on_batch([x_test, query, state])


class ResultSummary(
    collections.namedtuple("ResultSummary", ("shape", "dtype", "mean"))
):
    pass


def get_result_summary(x):
    if isinstance(x, np.ndarray):
        return ResultSummary(x.shape, x.dtype, x.mean())
    return x


@test_utils.run_all_in_graph_and_eager_modes
class AttentionWrapperTest(tf.test.TestCase, parameterized.TestCase):
    def assertAllCloseOrEqual(self, x, y, **kwargs):
        if isinstance(x, np.ndarray) or isinstance(x, float):
            return super().assertAllClose(x, y, atol=1e-3, **kwargs)
        else:
            self.assertAllEqual(x, y, **kwargs)

    def setUp(self):
        super().setUp()
        self.batch = 64
        self.units = 128
        self.encoder_timestep = 10
        self.encoder_dim = 256
        self.decoder_timestep = 12
        self.encoder_outputs = np.random.randn(
            self.batch, self.encoder_timestep, self.encoder_dim
        )
        self.encoder_sequence_length = np.random.randint(
            1, high=self.encoder_timestep, size=(self.batch,)
        ).astype(np.int32)
        self.decoder_inputs = np.random.randn(
            self.batch, self.decoder_timestep, self.units
        )
        self.decoder_sequence_length = np.random.randint(
            self.decoder_timestep, size=(self.batch,)
        ).astype(np.int32)

    def testCustomAttentionLayer(self):
        attention_mechanism = wrapper.LuongAttention(self.units)
        cell = tf.keras.layers.LSTMCell(self.units)
        attention_layer = tf.keras.layers.Dense(
            self.units * 2, use_bias=False, activation=tf.math.tanh
        )
        attention_wrapper = wrapper.AttentionWrapper(
            cell, attention_mechanism, attention_layer=attention_layer
        )
        with self.assertRaises(ValueError):
            # Should fail because the attention mechanism has not been
            # initialized.
            attention_wrapper.get_initial_state(batch_size=self.batch, dtype=tf.float32)
        attention_mechanism.setup_memory(
            self.encoder_outputs.astype(np.float32),
            memory_sequence_length=self.encoder_sequence_length,
        )
        initial_state = attention_wrapper.get_initial_state(
            batch_size=self.batch, dtype=tf.float32
        )
        self.assertEqual(initial_state.attention.shape[-1], self.units * 2)
        first_input = self.decoder_inputs[:, 0].astype(np.float32)
        output, _ = attention_wrapper(first_input, initial_state)
        self.assertEqual(output.shape[-1], self.units * 2)

    def _testWithAttention(
        self,
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=3,
        alignment_history=False,
        expected_final_alignment_history=None,
        attention_layer_size=6,
        attention_layer=None,
        create_query_layer=False,
        create_memory_layer=True,
        create_attention_kwargs=None,
    ):
        attention_layer_sizes = (
            [attention_layer_size] if attention_layer_size is not None else None
        )
        attention_layers = [attention_layer] if attention_layer is not None else None
        self._testWithMaybeMultiAttention(
            is_multi=False,
            create_attention_mechanisms=[create_attention_mechanism],
            expected_final_output=expected_final_output,
            expected_final_state=expected_final_state,
            attention_mechanism_depths=[attention_mechanism_depth],
            alignment_history=alignment_history,
            expected_final_alignment_history=expected_final_alignment_history,
            attention_layer_sizes=attention_layer_sizes,
            attention_layers=attention_layers,
            create_query_layer=create_query_layer,
            create_memory_layer=create_memory_layer,
            create_attention_kwargs=create_attention_kwargs,
        )

    def _testWithMaybeMultiAttention(
        self,
        is_multi,
        create_attention_mechanisms,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depths,
        alignment_history=False,
        expected_final_alignment_history=None,
        attention_layer_sizes=None,
        attention_layers=None,
        create_query_layer=False,
        create_memory_layer=True,
        create_attention_kwargs=None,
    ):
        # Allow is_multi to be True with a single mechanism to enable test for
        # passing in a single mechanism in a list.
        assert len(create_attention_mechanisms) == 1 or is_multi
        encoder_sequence_length = [3, 2, 3, 1, 1]
        decoder_sequence_length = [2, 0, 1, 2, 3]
        batch_size = 5
        encoder_max_time = 8
        decoder_max_time = 4
        input_depth = 7
        encoder_output_depth = 10
        cell_depth = 9
        create_attention_kwargs = create_attention_kwargs or {}

        if attention_layer_sizes is not None:
            # Compute sum of attention_layer_sizes. Use encoder_output_depth if
            # None.
            attention_depth = sum(
                attention_layer_size or encoder_output_depth
                for attention_layer_size in attention_layer_sizes
            )
        elif attention_layers is not None:
            # Compute sum of attention_layers output depth.
            attention_depth = sum(
                attention_layer.compute_output_shape(
                    [batch_size, cell_depth + encoder_output_depth]
                )
                .dims[-1]
                .value
                for attention_layer in attention_layers
            )
        else:
            attention_depth = encoder_output_depth * len(create_attention_mechanisms)

        decoder_inputs = np.random.randn(
            batch_size, decoder_max_time, input_depth
        ).astype(np.float32)
        encoder_outputs = np.random.randn(
            batch_size, encoder_max_time, encoder_output_depth
        ).astype(np.float32)

        attention_mechanisms = []
        for creator, depth in zip(
            create_attention_mechanisms, attention_mechanism_depths
        ):
            # Create a memory layer with deterministic initializer to avoid
            # randomness in the test between graph and eager.
            if create_query_layer:
                create_attention_kwargs["query_layer"] = tf.keras.layers.Dense(
                    depth, kernel_initializer="ones", use_bias=False
                )
            if create_memory_layer:
                create_attention_kwargs["memory_layer"] = tf.keras.layers.Dense(
                    depth, kernel_initializer="ones", use_bias=False
                )

            attention_mechanisms.append(
                creator(
                    units=depth,
                    memory=encoder_outputs,
                    memory_sequence_length=encoder_sequence_length,
                    **create_attention_kwargs,
                )
            )

        with self.cached_session(use_gpu=True):
            attention_layer_size = attention_layer_sizes
            attention_layer = attention_layers
            if not is_multi:
                if attention_layer_size is not None:
                    attention_layer_size = attention_layer_size[0]
                if attention_layer is not None:
                    attention_layer = attention_layer[0]
            cell = tf.keras.layers.LSTMCell(
                cell_depth,
                recurrent_activation="sigmoid",
                kernel_initializer="ones",
                recurrent_initializer="ones",
            )
            cell = wrapper.AttentionWrapper(
                cell,
                attention_mechanisms if is_multi else attention_mechanisms[0],
                attention_layer_size=attention_layer_size,
                alignment_history=alignment_history,
                attention_layer=attention_layer,
            )
            if cell._attention_layers is not None:
                for layer in cell._attention_layers:
                    layer.kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform(
                        seed=1337
                    )

            sampler = sampler_py.TrainingSampler()
            my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler)
            initial_state = cell.get_initial_state(
                dtype=tf.float32, batch_size=batch_size
            )
            final_outputs, final_state, _ = my_decoder(
                decoder_inputs,
                initial_state=initial_state,
                sequence_length=decoder_sequence_length,
            )

            self.assertIsInstance(final_outputs, basic_decoder.BasicDecoderOutput)
            self.assertIsInstance(final_state, wrapper.AttentionWrapperState)

            expected_time = (
                max(decoder_sequence_length) if tf.executing_eagerly() else None
            )
            self.assertEqual(
                (batch_size, expected_time, attention_depth),
                tuple(final_outputs.rnn_output.get_shape().as_list()),
            )
            self.assertEqual(
                (batch_size, expected_time),
                tuple(final_outputs.sample_id.get_shape().as_list()),
            )

            self.assertEqual(
                (batch_size, attention_depth),
                tuple(final_state.attention.get_shape().as_list()),
            )
            self.assertEqual(
                (batch_size, cell_depth),
                tuple(final_state.cell_state[0].get_shape().as_list()),
            )
            self.assertEqual(
                (batch_size, cell_depth),
                tuple(final_state.cell_state[1].get_shape().as_list()),
            )

            if alignment_history:
                if is_multi:
                    state_alignment_history = []
                    for history_array in final_state.alignment_history:
                        history = history_array.stack()
                        self.assertEqual(
                            (expected_time, batch_size, encoder_max_time),
                            tuple(history.get_shape().as_list()),
                        )
                        state_alignment_history.append(history)
                    state_alignment_history = tuple(state_alignment_history)
                else:
                    state_alignment_history = final_state.alignment_history.stack()
                    self.assertEqual(
                        (expected_time, batch_size, encoder_max_time),
                        tuple(state_alignment_history.get_shape().as_list()),
                    )
                tf.nest.assert_same_structure(
                    cell.state_size,
                    cell.get_initial_state(batch_size=batch_size, dtype=tf.float32),
                )
                # Remove the history from final_state for purposes of the
                # remainder of the tests.
                final_state = final_state._replace(
                    alignment_history=()
                )  # pylint: disable=protected-access
            else:
                state_alignment_history = ()

            self.evaluate(tf.compat.v1.global_variables_initializer())
            eval_result = self.evaluate(
                {
                    "final_outputs": final_outputs,
                    "final_state": final_state,
                    "state_alignment_history": state_alignment_history,
                }
            )

            final_output_info = tf.nest.map_structure(
                get_result_summary, eval_result["final_outputs"]
            )
            final_state_info = tf.nest.map_structure(
                get_result_summary, eval_result["final_state"]
            )
            print("final_output_info: ", final_output_info)
            print("final_state_info: ", final_state_info)

            tf.nest.map_structure(
                self.assertAllCloseOrEqual, expected_final_output, final_output_info
            )
            tf.nest.map_structure(
                self.assertAllCloseOrEqual, expected_final_state, final_state_info
            )
            # by default, the wrapper emits attention as output
            if alignment_history:
                final_alignment_history_info = tf.nest.map_structure(
                    get_result_summary, eval_result["state_alignment_history"]
                )
                print("final_alignment_history_info: ", final_alignment_history_info)
                tf.nest.map_structure(
                    self.assertAllCloseOrEqual,
                    # outputs are batch major but the stacked TensorArray is
                    # time major
                    expected_final_alignment_history,
                    final_alignment_history_info,
                )

    @parameterized.parameters([np.float32, np.float64])
    def testBahdanauNormalizedDType(self, dtype):
        encoder_outputs = self.encoder_outputs.astype(dtype)
        decoder_inputs = self.decoder_inputs.astype(dtype)
        attention_mechanism = wrapper.BahdanauAttention(
            units=self.units,
            memory=encoder_outputs,
            memory_sequence_length=self.encoder_sequence_length,
            normalize=True,
            dtype=dtype,
        )
        cell = tf.keras.layers.LSTMCell(
            self.units, recurrent_activation="sigmoid", dtype=dtype
        )
        cell = wrapper.AttentionWrapper(cell, attention_mechanism, dtype=dtype)

        sampler = sampler_py.TrainingSampler()
        my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler, dtype=dtype)

        final_outputs, final_state, _ = my_decoder(
            decoder_inputs,
            initial_state=cell.get_initial_state(batch_size=self.batch, dtype=dtype),
            sequence_length=self.decoder_sequence_length,
        )
        self.assertIsInstance(final_outputs, basic_decoder.BasicDecoderOutput)
        self.assertEqual(final_outputs.rnn_output.dtype, dtype)
        self.assertIsInstance(final_state, wrapper.AttentionWrapperState)

    @parameterized.parameters([np.float32, np.float64])
    def testLuongScaledDType(self, dtype):
        # Test case for GitHub issue 18099
        encoder_outputs = self.encoder_outputs.astype(dtype)
        decoder_inputs = self.decoder_inputs.astype(dtype)
        attention_mechanism = wrapper.LuongAttention(
            units=self.units,
            memory=encoder_outputs,
            memory_sequence_length=self.encoder_sequence_length,
            scale=True,
            dtype=dtype,
        )
        cell = tf.keras.layers.LSTMCell(
            self.units, recurrent_activation="sigmoid", dtype=dtype
        )
        cell = wrapper.AttentionWrapper(cell, attention_mechanism, dtype=dtype)

        sampler = sampler_py.TrainingSampler()
        my_decoder = basic_decoder.BasicDecoder(cell=cell, sampler=sampler, dtype=dtype)

        final_outputs, final_state, _ = my_decoder(
            decoder_inputs,
            initial_state=cell.get_initial_state(batch_size=self.batch, dtype=dtype),
            sequence_length=self.decoder_sequence_length,
        )
        self.assertIsInstance(final_outputs, basic_decoder.BasicDecoderOutput)
        self.assertEqual(final_outputs.rnn_output.dtype, dtype)
        self.assertIsInstance(final_state, wrapper.AttentionWrapperState)

    def testBahdanauNotNormalized(self):
        create_attention_mechanism = wrapper.BahdanauAttention
        create_attention_kwargs = {"kernel_initializer": "ones"}
        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype(np.float32), mean=-0.003204414
            ),
            sample_id=ResultSummary(shape=(5, 3), dtype=np.dtype(np.int32), mean=3.2),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(
                    shape=(5, 9), dtype=np.dtype(np.float32), mean=0.40868404
                ),
                ResultSummary(
                    shape=(5, 9), dtype=np.dtype(np.float32), mean=0.89017969
                ),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype(np.float32), mean=0.041453815
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype(np.float32), mean=0.125
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype(np.float32), mean=0.125
            ),
            alignment_history=(),
        )
        expected_final_alignment_history = ResultSummary(
            shape=(3, 5, 8), dtype=np.dtype(np.float32), mean=0.125
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            alignment_history=True,
            create_query_layer=True,
            expected_final_alignment_history=expected_final_alignment_history,
            create_attention_kwargs=create_attention_kwargs,
        )

    def testBahdanauNormalized(self):
        create_attention_mechanism = wrapper.BahdanauAttention
        create_attention_kwargs = {"kernel_initializer": "ones", "normalize": True}

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=-0.008089137
            ),
            sample_id=ResultSummary(shape=(5, 3), dtype=np.dtype("int32"), mean=2.8),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.49166861),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.01068615),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=0.042427111
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            alignment_history=(),
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            create_query_layer=True,
            create_attention_kwargs=create_attention_kwargs,
        )

    def testLuongNotNormalized(self):
        create_attention_mechanism = wrapper.LuongAttention

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=-0.06124732
            ),
            sample_id=ResultSummary(
                shape=(5, 3), dtype=np.dtype("int32"), mean=2.73333333
            ),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.52021580),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.0964939),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=-0.0318060
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            alignment_history=(),
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            attention_mechanism_depth=9,
        )

    def testLuongScaled(self):
        create_attention_mechanism = wrapper.LuongAttention
        create_attention_kwargs = {"scale": True}

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=-0.06124732
            ),
            sample_id=ResultSummary(
                shape=(5, 3), dtype=np.dtype("int32"), mean=2.73333333
            ),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.52021580),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.0964939),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=-0.0318060
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            alignment_history=(),
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            attention_mechanism_depth=9,
            create_attention_kwargs=create_attention_kwargs,
        )

    def testNotUseAttentionLayer(self):
        create_attention_mechanism = wrapper.BahdanauAttention
        create_attention_kwargs = {"kernel_initializer": "ones"}

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 10), dtype=np.dtype("float32"), mean=0.078317143
            ),
            sample_id=ResultSummary(shape=(5, 3), dtype=np.dtype("int32"), mean=4.2),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.89382392),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.722382),
            ],
            attention=ResultSummary(
                shape=(5, 10), dtype=np.dtype("float32"), mean=0.026356646
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.125
            ),
            alignment_history=(),
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            attention_layer_size=None,
            create_query_layer=True,
            create_attention_kwargs=create_attention_kwargs,
        )

    def testBahdanauMonotonicNotNormalized(self):
        create_attention_mechanism = wrapper.BahdanauMonotonicAttention
        create_attention_kwargs = {"kernel_initializer": "ones"}

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=-0.009921653
            ),
            sample_id=ResultSummary(
                shape=(5, 3), dtype=np.dtype("int32"), mean=3.13333333
            ),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.44612807),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.95786464),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=0.038682378
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.09778417
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.09778417
            ),
            alignment_history=(),
        )
        expected_final_alignment_history = ResultSummary(
            shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.10261579603
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            alignment_history=True,
            expected_final_alignment_history=expected_final_alignment_history,
            create_query_layer=True,
            create_attention_kwargs=create_attention_kwargs,
        )

    def testBahdanauMonotonicNormalized(self):
        create_attention_mechanism = wrapper.BahdanauMonotonicAttention
        create_attention_kwargs = {"kernel_initializer": "ones", "normalize": True}
        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.007140680
            ),
            sample_id=ResultSummary(
                shape=(5, 3), dtype=np.dtype("int32"), mean=3.26666666
            ),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.47012400),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.0249618),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=0.068432882
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.0615656
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.0615656
            ),
            alignment_history=(),
        )
        expected_final_alignment_history = ResultSummary(
            shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.07909643
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            alignment_history=True,
            expected_final_alignment_history=expected_final_alignment_history,
            create_query_layer=True,
            create_attention_kwargs=create_attention_kwargs,
        )

    def testLuongMonotonicNotNormalized(self):
        create_attention_mechanism = wrapper.LuongMonotonicAttention

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.003664831
            ),
            sample_id=ResultSummary(
                shape=(5, 3), dtype=np.dtype("int32"), mean=3.06666666
            ),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.54318606),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.12592840),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=0.059128221
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.05112994
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.05112994
            ),
            alignment_history=(),
        )
        expected_final_alignment_history = ResultSummary(
            shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.06994973868
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            attention_mechanism_depth=9,
            alignment_history=True,
            expected_final_alignment_history=expected_final_alignment_history,
        )

    def testLuongMonotonicScaled(self):
        create_attention_mechanism = wrapper.LuongMonotonicAttention
        create_attention_kwargs = {"scale": True}

        expected_final_output = basic_decoder.BasicDecoderOutput(
            rnn_output=ResultSummary(
                shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.003664831
            ),
            sample_id=ResultSummary(
                shape=(5, 3), dtype=np.dtype("int32"), mean=3.06666666
            ),
        )
        expected_final_state = wrapper.AttentionWrapperState(
            cell_state=[
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=0.54318606),
                ResultSummary(shape=(5, 9), dtype=np.dtype("float32"), mean=1.12592840),
            ],
            attention=ResultSummary(
                shape=(5, 6), dtype=np.dtype("float32"), mean=0.059128221
            ),
            alignments=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.05112994
            ),
            attention_state=ResultSummary(
                shape=(5, 8), dtype=np.dtype("float32"), mean=0.05112994
            ),
            alignment_history=(),
        )
        expected_final_alignment_history = ResultSummary(
            shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.06994973868
        )

        self._testWithAttention(
            create_attention_mechanism,
            expected_final_output,
            expected_final_state,
            attention_mechanism_depth=9,
            alignment_history=True,
            expected_final_alignment_history=expected_final_alignment_history,
            create_attention_kwargs=create_attention_kwargs,
        )

    def test_attention_state_with_keras_rnn(self):
        # See https://github.com/tensorflow/addons/issues/1095.
        cell = tf.keras.layers.LSTMCell(8)

        mechanism = wrapper.LuongAttention(units=8, memory=tf.ones((2, 4, 8)))

        cell = wrapper.AttentionWrapper(cell=cell, attention_mechanism=mechanism)

        layer = tf.keras.layers.RNN(cell)
        _ = layer(inputs=tf.ones((2, 4, 8)))

        # Make sure the explicit initial_state also works.
        initial_state = cell.get_initial_state(batch_size=2, dtype=tf.float32)
        _ = layer(inputs=tf.ones((2, 4, 8)), initial_state=initial_state)

    def test_attention_state_with_variable_length_input(self):
        cell = tf.keras.layers.LSTMCell(3)
        mechanism = wrapper.LuongAttention(units=3)
        cell = wrapper.AttentionWrapper(cell, mechanism)

        var_len = tf.random.uniform(shape=(), minval=2, maxval=10, dtype=tf.int32)
        lengths = tf.random.uniform(
            shape=(var_len,), minval=1, maxval=var_len + 1, dtype=tf.int32
        )
        data = tf.ones(shape=(var_len, var_len, 3))
        mask = tf.sequence_mask(lengths, maxlen=var_len)

        mechanism.setup_memory(data)
        layer = tf.keras.layers.RNN(cell)

        _ = layer(data, mask=mask)
