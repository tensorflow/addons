# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A basic decoder that may sample to generate the next input."""

import collections

import tensorflow as tf

from tensorflow_addons.seq2seq import decoder
from tensorflow_addons.seq2seq import sampler as sampler_py
from tensorflow_addons.utils import keras_utils

from typeguard import typechecked
from typing import Optional


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))
):
    """Outputs of a `tfa.seq2seq.BasicDecoder` step.

    Attributes:
      rnn_output: The output for this step. If the `output_layer` argument
         of `tfa.seq2seq.BasicDecoder` was set, it is the output of this layer, otherwise it
         is the output of the RNN cell.
      sample_id: The token IDs sampled for this step, as returned by the
        `sampler` instance passed to `tfa.seq2seq.BasicDecoder`.
    """

    pass


class BasicDecoder(decoder.BaseDecoder):
    """Basic sampling decoder for training and inference.

    The `tfa.seq2seq.Sampler` instance passed as argument is responsible to sample from
    the output distribution and produce the input for the next decoding step. The decoding
    loop is implemented by the decoder in its `__call__` method.

    Example using `tfa.seq2seq.TrainingSampler` for training:

    >>> batch_size = 4
    >>> max_time = 7
    >>> hidden_size = 32
    >>> embedding_size = 48
    >>> input_vocab_size = 128
    >>> output_vocab_size = 64
    >>>
    >>> embedding_layer = tf.keras.layers.Embedding(input_vocab_size, embedding_size)
    >>> decoder_cell = tf.keras.layers.LSTMCell(hidden_size)
    >>> sampler = tfa.seq2seq.TrainingSampler()
    >>> output_layer = tf.keras.layers.Dense(output_vocab_size)
    >>>
    >>> decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer)
    >>>
    >>> input_ids = tf.random.uniform(
    ...     [batch_size, max_time], maxval=input_vocab_size, dtype=tf.int64)
    >>> input_lengths = tf.fill([batch_size], max_time)
    >>> input_tensors = embedding_layer(input_ids)
    >>> initial_state = decoder_cell.get_initial_state(input_tensors)
    >>>
    >>> output, state, lengths = decoder(
    ...     input_tensors, sequence_length=input_lengths, initial_state=initial_state)
    >>>
    >>> logits = output.rnn_output
    >>> logits.shape
    TensorShape([4, 7, 64])

    Example using `tfa.seq2seq.GreedyEmbeddingSampler` for inference:

    >>> sampler = tfa.seq2seq.GreedyEmbeddingSampler(embedding_layer)
    >>> decoder = tfa.seq2seq.BasicDecoder(
    ...     decoder_cell, sampler, output_layer, maximum_iterations=10)
    >>>
    >>> initial_state = decoder_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
    >>> start_tokens = tf.fill([batch_size], 1)
    >>> end_token = 2
    >>>
    >>> output, state, lengths = decoder(
    ...     None, start_tokens=start_tokens, end_token=end_token, initial_state=initial_state)
    >>>
    >>> output.sample_id.shape
    TensorShape([4, 10])
    """

    @typechecked
    def __init__(
        self,
        cell: tf.keras.layers.Layer,
        sampler: sampler_py.Sampler,
        output_layer: Optional[tf.keras.layers.Layer] = None,
        **kwargs,
    ):
        """Initialize BasicDecoder.

        Args:
          cell: A layer that implements the `tf.keras.layers.AbstractRNNCell`
            interface.
          sampler: A `tfa.seq2seq.Sampler` instance.
          output_layer: (Optional) An instance of `tf.keras.layers.Layer`, i.e.,
            `tf.keras.layers.Dense`. Optional layer to apply to the RNN output
             prior to storing the result or sampling.
          **kwargs: Other keyword arguments of `tfa.seq2seq.BaseDecoder`.
        """
        keras_utils.assert_like_rnncell("cell", cell)
        self.cell = cell
        self.sampler = sampler
        self.output_layer = output_layer
        super().__init__(**kwargs)

    def initialize(self, inputs, initial_state=None, **kwargs):
        """Initialize the decoder."""
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        self._cell_dtype = tf.nest.flatten(initial_state)[0].dtype
        return self.sampler.initialize(inputs, **kwargs) + (initial_state,)

    @property
    def batch_size(self):
        return self.sampler.batch_size

    def _rnn_output_size(self):
        size = tf.TensorShape(self.cell.output_size)
        if self.output_layer is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = tf.nest.map_structure(
                lambda s: tf.TensorShape([None]).concatenate(s), size
            )
            layer_output_shape = self.output_layer.compute_output_shape(
                output_shape_with_unknown_batch
            )
            return tf.nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(), sample_id=self.sampler.sample_ids_shape
        )

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = self._cell_dtype
        return BasicDecoderOutput(
            tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self.sampler.sample_ids_dtype,
        )

    def step(self, time, inputs, state, training=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          training: Python boolean.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        cell_outputs, cell_state = self.cell(inputs, state, training=training)
        cell_state = tf.nest.pack_sequence_as(state, tf.nest.flatten(cell_state))
        if self.output_layer is not None:
            cell_outputs = self.output_layer(cell_outputs)
        sample_ids = self.sampler.sample(
            time=time, outputs=cell_outputs, state=cell_state
        )
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time, outputs=cell_outputs, state=cell_state, sample_ids=sample_ids
        )
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)
