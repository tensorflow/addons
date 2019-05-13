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
"""A class of Decoders that may sample to generate the next input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow_addons.seq2seq import decoder
from tensorflow_addons.seq2seq import sampler as sampler_py

# TODO: Find public API alternatives to this
from tensorflow.python.ops import rnn_cell_impl


class BasicDecoderOutput(
        collections.namedtuple("BasicDecoderOutput",
                               ("rnn_output", "sample_id"))):
    pass


class BasicDecoder(decoder.BaseDecoder):
    """Basic sampling decoder."""

    def __init__(self, cell, sampler, output_layer=None, **kwargs):
        """Initialize BasicDecoder.

        Args:
          cell: An `RNNCell` instance.
          sampler: A `Sampler` instance.
          output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`. Optional layer to apply to the RNN output prior
             to storing the result or sampling.
          **kwargs: Other keyward arguments for layer creation.

        Raises:
          TypeError: if `cell`, `helper` or `output_layer` have an incorrect
          type.
        """
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if not isinstance(sampler, sampler_py.Sampler):
            raise TypeError(
                "sampler must be a Sampler, received: %s" % (sampler,))
        if (output_layer is not None
                and not isinstance(output_layer, tf.keras.layers.Layer)):
            raise TypeError(
                "output_layer must be a Layer, received: %s" % (output_layer,))
        self.cell = cell
        self.sampler = sampler
        self.output_layer = output_layer
        super(BasicDecoder, self).__init__(**kwargs)

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
                lambda s: tf.TensorShape([None]).concatenate(s), size)
            layer_output_shape = self.output_layer.compute_output_shape(
                output_shape_with_unknown_batch)
            return tf.nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def output_size(self):
        # Return the cell output and the id
        return BasicDecoderOutput(
            rnn_output=self._rnn_output_size(),
            sample_id=self.sampler.sample_ids_shape)

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and the sample_ids_dtype from the helper.
        dtype = self._cell_dtype
        return BasicDecoderOutput(
            tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
            self.sampler.sample_ids_dtype)

    def step(self, time, inputs, state):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        cell_outputs, cell_state = self.cell(inputs, state)
        if self.output_layer is not None:
            cell_outputs = self.output_layer(cell_outputs)
        sample_ids = self.sampler.sample(
            time=time, outputs=cell_outputs, state=cell_state)
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time,
            outputs=cell_outputs,
            state=cell_state,
            sample_ids=sample_ids)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return (outputs, next_state, next_inputs, finished)
