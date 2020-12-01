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
"""Base classes and functions for dynamic decoding."""

import abc

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike
from typeguard import typechecked
from typing import Any, Optional, Tuple, Union

# TODO: Find public API alternatives to these
from tensorflow.python.ops import control_flow_util


class Decoder(metaclass=abc.ABCMeta):
    """An RNN Decoder abstract interface object.

    Concepts used by this interface:
    - `inputs`: (structure of) tensors and TensorArrays that is passed as input
      to the RNN cell composing the decoder, at each time step.
    - `state`: (structure of) tensors and TensorArrays that is passed to the
      RNN cell instance as the state.
    - `finished`: boolean tensor telling whether each sequence in the batch is
      finished.
    - `training`: boolean whether it should behave in training mode or in
      inference mode.
    - `outputs`: instance of `tfa.seq2seq.BasicDecoderOutput`. Result of the decoding, at
      each time step.
    """

    @property
    def batch_size(self):
        """The batch size of input values."""
        raise NotImplementedError

    @property
    def output_size(self):
        """A (possibly nested tuple of...) integer[s] or `TensorShape`
        object[s]."""
        raise NotImplementedError

    @property
    def output_dtype(self):
        """A (possibly nested tuple of...) dtype[s]."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self, name=None):
        """Called before any decoding iterations.

        This methods must compute initial input values and initial state.

        Args:
          name: Name scope for any created operations.

        Returns:
          `(finished, initial_inputs, initial_state)`: initial values of
          'finished' flags, inputs and state.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, time, inputs, state, training=None, name=None):
        """Called per step of decoding (but only once for dynamic decoding).

        Args:
          time: Scalar `int32` tensor. Current step number.
          inputs: RNN cell input (possibly nested tuple of) tensor[s] for this
            time step.
          state: RNN cell state (possibly nested tuple of) tensor[s] from
            previous time step.
          training: Python boolean. Indicates whether the layer should behave
            in training  mode or in inference mode. Only relevant
            when `dropout` or `recurrent_dropout` is used.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`: `outputs` is an
          object containing the decoder output, `next_state` is a (structure
          of) state tensors and TensorArrays, `next_inputs` is the tensor that
          should be used as input for the next step, `finished` is a boolean
          tensor telling whether the sequence is complete, for each sequence in
          the batch.
        """
        raise NotImplementedError

    def finalize(self, outputs, final_state, sequence_lengths):
        raise NotImplementedError

    @property
    def tracks_own_finished(self):
        """Describes whether the Decoder keeps track of finished states.

        Most decoders will emit a true/false `finished` value independently
        at each time step.  In this case, the `tfa.seq2seq.dynamic_decode` function keeps
        track of which batch entries are already finished, and performs a
        logical OR to insert new batches to the finished set.

        Some decoders, however, shuffle batches / beams between time steps and
        `tfa.seq2seq.dynamic_decode` will mix up the finished state across these entries
        because it does not track the reshuffle across time steps. In this
        case, it is up to the decoder to declare that it will keep track of its
        own finished state by setting this property to `True`.

        Returns:
          Python bool.
        """
        return False


class BaseDecoder(tf.keras.layers.Layer):
    """An RNN Decoder that is based on a Keras layer.

    Concepts used by this interface:
    - `inputs`: (structure of) Tensors and TensorArrays that is passed as input
      to the RNN cell composing the decoder, at each time step.
    - `state`: (structure of) Tensors and TensorArrays that is passed to the
      RNN cell instance as the state.
    - `memory`: tensor that is usually the full output of the encoder, which
      will be used for the attention wrapper for the RNN cell.
    - `finished`: boolean tensor telling whether each sequence in the batch is
      finished.
    - `training`: boolean whether it should behave in training mode or in
      inference mode.
    - `outputs`: instance of `tfa.seq2seq.BasicDecoderOutput`. Result of the decoding, at
      each time step.
    """

    @typechecked
    def __init__(
        self,
        output_time_major: bool = False,
        impute_finished: bool = False,
        maximum_iterations: Optional[TensorLike] = None,
        parallel_iterations: int = 32,
        swap_memory: bool = False,
        **kwargs,
    ):
        self.output_time_major = output_time_major
        self.impute_finished = impute_finished
        self.maximum_iterations = maximum_iterations
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory
        super().__init__(**kwargs)

    def call(self, inputs, initial_state=None, training=None, **kwargs):
        init_kwargs = kwargs
        init_kwargs["initial_state"] = initial_state
        return dynamic_decode(
            self,
            output_time_major=self.output_time_major,
            impute_finished=self.impute_finished,
            maximum_iterations=self.maximum_iterations,
            parallel_iterations=self.parallel_iterations,
            swap_memory=self.swap_memory,
            training=training,
            decoder_init_input=inputs,
            decoder_init_kwargs=init_kwargs,
        )

    @property
    def batch_size(self):
        """The batch size of input values."""
        raise NotImplementedError

    @property
    def output_size(self):
        """A (possibly nested tuple of...) integer[s] or `TensorShape`
        object[s]."""
        raise NotImplementedError

    @property
    def output_dtype(self):
        """A (possibly nested tuple of...) dtype[s]."""
        raise NotImplementedError

    def initialize(self, inputs, initial_state=None, **kwargs):
        """Called before any decoding iterations.

        This methods must compute initial input values and initial state.

        Args:
          inputs: (structure of) tensors that contains the input for the
            decoder. In the normal case, it's a tensor with shape
            [batch, timestep, embedding].
          initial_state: (structure of) tensors that contains the initial state
            for the RNN cell.
          **kwargs: Other arguments that are passed in from layer.call()
            method. It could contains item like input `sequence_length`, or
            masking for input.

        Returns:
          `(finished, initial_inputs, initial_state)`: initial values of
          'finished' flags, inputs and state.
        """
        raise NotImplementedError

    def step(self, time, inputs, state, training):
        """Called per step of decoding (but only once for dynamic decoding).

        Args:
          time: Scalar `int32` tensor. Current step number.
          inputs: RNN cell input (possibly nested tuple of) tensor[s] for this
            time step.
          state: RNN cell state (possibly nested tuple of) tensor[s] from
            previous time step.
          training: Python boolean. Indicates whether the layer should
            behave in training mode or in inference mode.

        Returns:
          `(outputs, next_state, next_inputs, finished)`: `outputs` is an
          object containing the decoder output, `next_state` is a
          (structure of) state tensors and TensorArrays, `next_inputs` is the
          tensor that should be used as input for the next step, `finished` is
          a boolean tensor telling whether the sequence is complete, for each
          sequence in the batch.
        """
        raise NotImplementedError

    def finalize(self, outputs, final_state, sequence_lengths):
        raise NotImplementedError

    @property
    def tracks_own_finished(self):
        """Describes whether the Decoder keeps track of finished states.

        Most decoders will emit a true/false `finished` value independently
        at each time step.  In this case, the `tfa.seq2seq.dynamic_decode` function keeps
        track of which batch entries are already finished, and performs a
        logical OR to insert new batches to the finished set.

        Some decoders, however, shuffle batches / beams between time steps and
        `tfa.seq2seq.dynamic_decode` will mix up the finished state across these entries
        because it does not track the reshuffle across time steps. In this
        case, it is up to the decoder to declare that it will keep track of its
        own finished state by setting this property to `True`.

        Returns:
          Python bool.
        """
        return False

    # TODO(scottzhu): Add build/get_config/from_config and other layer methods.


@typechecked
def dynamic_decode(
    decoder: Union[Decoder, BaseDecoder],
    output_time_major: bool = False,
    impute_finished: bool = False,
    maximum_iterations: Optional[TensorLike] = None,
    parallel_iterations: int = 32,
    swap_memory: bool = False,
    training: Optional[bool] = None,
    scope: Optional[str] = None,
    enable_tflite_convertible: bool = False,
    **kwargs,
) -> Tuple[Any, Any, Any]:
    """Runs dynamic decoding with a decoder.

    Calls `initialize()` once and `step()` repeatedly on the decoder object.

    Args:
      decoder: A `tfa.seq2seq.Decoder` or `tfa.seq2seq.BaseDecoder` instance.
      output_time_major: Python boolean.  Default: `False` (batch major). If
        `True`, outputs are returned as time major tensors (this mode is
        faster). Otherwise, outputs are returned as batch major tensors (this
        adds extra time to the computation).
      impute_finished: Python boolean.  If `True`, then states for batch
        entries which are marked as finished get copied through and the
        corresponding outputs get zeroed out.  This causes some slowdown at
        each time step, but ensures that the final state and outputs have
        the correct values and that backprop ignores time steps that were
        marked as finished.
      maximum_iterations: A strictly positive `int32` scalar, the maximum
         allowed number of decoding steps. Default is `None` (decode until the
         decoder is fully done).
      parallel_iterations: Argument passed to `tf.while_loop`.
      swap_memory: Argument passed to `tf.while_loop`.
      training: Python boolean. Indicates whether the layer should behave
          in training  mode or in inference mode. Only relevant
          when `dropout` or `recurrent_dropout` is used.
      scope: Optional name scope to use.
      enable_tflite_convertible: Python boolean. If `True`, then the variables
        of `TensorArray` become of 1-D static shape. Also zero pads in the
        output tensor will be discarded. Default: `False`.
      **kwargs: dict, other keyword arguments for dynamic_decode. It might
        contain arguments for `BaseDecoder` to initialize, which takes all
        tensor inputs during call().

    Returns:
      `(final_outputs, final_state, final_sequence_lengths)`.

    Raises:
      ValueError: if `maximum_iterations` is provided but is not a scalar.
    """
    with tf.name_scope(scope or "decoder"):
        is_xla = (
            not tf.executing_eagerly()
            and control_flow_util.GraphOrParentsInXlaContext(
                tf.compat.v1.get_default_graph()
            )
        )

        if maximum_iterations is not None:
            maximum_iterations = tf.convert_to_tensor(
                maximum_iterations, dtype=tf.int32, name="maximum_iterations"
            )
            if maximum_iterations.shape.ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")
            tf.debugging.assert_greater(
                maximum_iterations,
                0,
                message="maximum_iterations should be greater than 0",
            )
        elif is_xla:
            raise ValueError("maximum_iterations is required for XLA compilation.")

        if isinstance(decoder, Decoder):
            initial_finished, initial_inputs, initial_state = decoder.initialize()
        else:
            # For BaseDecoder that takes tensor inputs during call.
            decoder_init_input = kwargs.pop("decoder_init_input", None)
            decoder_init_kwargs = kwargs.pop("decoder_init_kwargs", {})
            initial_finished, initial_inputs, initial_state = decoder.initialize(
                decoder_init_input, **decoder_init_kwargs
            )

        if enable_tflite_convertible:
            # Assume the batch_size = 1 for inference.
            # So we can change 2-D TensorArray into 1-D by reshaping it.
            tf.debugging.assert_equal(
                decoder.batch_size,
                1,
                message="TFLite conversion requires a batch size of 1",
            )
            zero_outputs = tf.nest.map_structure(
                lambda shape, dtype: tf.reshape(
                    tf.zeros(_prepend_batch(decoder.batch_size, shape), dtype=dtype),
                    [-1],
                ),
                decoder.output_size,
                decoder.output_dtype,
            )
        else:
            zero_outputs = tf.nest.map_structure(
                lambda shape, dtype: tf.zeros(
                    _prepend_batch(decoder.batch_size, shape), dtype=dtype
                ),
                decoder.output_size,
                decoder.output_dtype,
            )

        if maximum_iterations is not None:
            initial_finished = tf.logical_or(initial_finished, 0 >= maximum_iterations)
        initial_sequence_lengths = tf.zeros_like(initial_finished, dtype=tf.int32)
        initial_time = tf.constant(0, dtype=tf.int32)

        def _shape(batch_size, from_shape):
            if not isinstance(from_shape, tf.TensorShape) or from_shape.ndims == 0:
                return None
            else:
                batch_size = tf.get_static_value(
                    tf.convert_to_tensor(batch_size, name="batch_size")
                )
                return tf.TensorShape([batch_size]).concatenate(from_shape)

        dynamic_size = maximum_iterations is None or not is_xla
        # The dynamic shape `TensorArray` is not allowed in TFLite yet.
        dynamic_size = dynamic_size and (not enable_tflite_convertible)

        def _create_ta(s, d):
            if enable_tflite_convertible:
                # TFLite requires 1D element_shape.
                if isinstance(s, tf.TensorShape) and s.ndims == 0:
                    s = (1,)
                element_shape = s
            else:
                element_shape = _shape(decoder.batch_size, s)
            return tf.TensorArray(
                dtype=d,
                size=0 if dynamic_size else maximum_iterations,
                dynamic_size=dynamic_size,
                element_shape=element_shape,
            )

        initial_outputs_ta = tf.nest.map_structure(
            _create_ta, decoder.output_size, decoder.output_dtype
        )

        def condition(
            unused_time,
            unused_outputs_ta,
            unused_state,
            unused_inputs,
            finished,
            unused_sequence_lengths,
        ):
            return tf.logical_not(tf.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
            """Internal while_loop body.

            Args:
              time: scalar int32 tensor.
              outputs_ta: structure of TensorArray.
              state: (structure of) state tensors and TensorArrays.
              inputs: (structure of) input tensors.
              finished: bool tensor (keeping track of what's finished).
              sequence_lengths: int32 tensor (keeping track of time of finish).

            Returns:
              `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
                next_sequence_lengths)`.
              ```
            """
            (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(
                time, inputs, state, training
            )
            decoder_state_sequence_lengths = False
            if decoder.tracks_own_finished:
                next_finished = decoder_finished
                lengths = getattr(decoder_state, "lengths", None)
                if lengths is not None:
                    # sequence lengths are provided by decoder_state.lengths;
                    # overwrite our sequence lengths.
                    decoder_state_sequence_lengths = True
                    sequence_lengths = tf.cast(lengths, tf.int32)
            else:
                next_finished = tf.logical_or(decoder_finished, finished)

            if decoder_state_sequence_lengths:
                # Just pass something through the loop; at the next iteration
                # we'll pull the sequence lengths from the decoder_state again.
                next_sequence_lengths = sequence_lengths
            else:
                next_sequence_lengths = tf.where(
                    tf.logical_not(finished),
                    tf.fill(tf.shape(sequence_lengths), time + 1),
                    sequence_lengths,
                )

            tf.nest.assert_same_structure(state, decoder_state)
            tf.nest.assert_same_structure(outputs_ta, next_outputs)
            tf.nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:

                def zero_out_finished(out, zero):
                    if finished.shape.rank < zero.shape.rank:
                        broadcast_finished = tf.broadcast_to(
                            tf.expand_dims(finished, axis=-1), zero.shape
                        )
                        return tf.where(broadcast_finished, zero, out)
                    else:
                        return tf.where(finished, zero, out)

                emit = tf.nest.map_structure(
                    zero_out_finished, next_outputs, zero_outputs
                )
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tf.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = new.shape.ndims == 0
                if not pass_through:
                    broadcast_finished = tf.broadcast_to(
                        tf.expand_dims(finished, axis=-1), new.shape
                    )
                    return tf.where(broadcast_finished, cur, new)
                else:
                    return new

            if impute_finished:
                next_state = tf.nest.map_structure(
                    _maybe_copy_state, decoder_state, state
                )
            else:
                next_state = decoder_state

            if enable_tflite_convertible:
                # Reshape to 1-D.
                emit = tf.nest.map_structure(lambda x: tf.reshape(x, [-1]), emit)

            outputs_ta = tf.nest.map_structure(
                lambda ta, out: ta.write(time, out), outputs_ta, emit
            )
            return (
                time + 1,
                outputs_ta,
                next_state,
                next_inputs,
                next_finished,
                next_sequence_lengths,
            )

        res = tf.while_loop(
            condition,
            body,
            loop_vars=(
                initial_time,
                initial_outputs_ta,
                initial_state,
                initial_inputs,
                initial_finished,
                initial_sequence_lengths,
            ),
            parallel_iterations=parallel_iterations,
            maximum_iterations=maximum_iterations,
            swap_memory=swap_memory,
        )

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]

        final_outputs = tf.nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths
            )
        except NotImplementedError:
            pass

        if not output_time_major:
            if enable_tflite_convertible:
                # Reshape the output to the original shape.
                def _restore_batch(x):
                    return tf.expand_dims(x, [1])

                final_outputs = tf.nest.map_structure(_restore_batch, final_outputs)

            final_outputs = tf.nest.map_structure(_transpose_batch_time, final_outputs)

    return final_outputs, final_state, final_sequence_lengths


def _prepend_batch(batch_size, shape):
    """Prepends the batch dimension to the shape.

    If the batch_size value is known statically, this function returns a
    TensorShape, otherwise a Tensor.
    """
    if isinstance(batch_size, tf.Tensor):
        static_batch_size = tf.get_static_value(batch_size)
    else:
        static_batch_size = batch_size
    if static_batch_size is None:
        return tf.concat(([batch_size], shape), axis=0)
    return [static_batch_size] + shape


def _transpose_batch_time(tensor):
    """Transposes the batch and time dimension of tensor if its rank is at
    least 2."""
    shape = tensor.shape
    if shape.rank is not None and shape.rank < 2:
        return tensor
    perm = tf.concat(([1, 0], tf.range(2, tf.rank(tensor))), axis=0)
    return tf.transpose(tensor, perm)
