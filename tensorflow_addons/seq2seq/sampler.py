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
"""A library of sampler for use with SamplingDecoders."""

import abc

import tensorflow as tf
from tensorflow_addons.seq2seq import decoder
from tensorflow_addons.utils.types import Initializer, TensorLike
from typeguard import typechecked
from typing import Callable, Optional
from tensorflow_addons.utils import types

_transpose_batch_time = decoder._transpose_batch_time


class Sampler(metaclass=abc.ABCMeta):
    """Interface for implementing sampling in seq2seq decoders.

    Sampler instances are used by `BasicDecoder`. The normal usage of a sampler
    is like below:

    ```python
    sampler = Sampler(init_args)
    (initial_finished, initial_inputs) = sampler.initialize(input_tensors)
    cell_input = initial_inputs
    cell_state = cell.get_initial_state(...)
    for time_step in tf.range(max_output_length):
        cell_output, cell_state = cell(cell_input, cell_state)
        sample_ids = sampler.sample(time_step, cell_output, cell_state)
        (finished, cell_input, cell_state) = sampler.next_inputs(
            time_step, cell_output, cell_state, sample_ids)
        if tf.reduce_all(finished):
            break
    ```

    Note that the input_tensors should not be fed to the Sampler as __init__()
    parameters. Instead, they should be fed by decoders via initialize().
    """

    @abc.abstractmethod
    def initialize(self, inputs, **kwargs):
        """initialize the sampler with the input tensors.

        This method must be invoked exactly once before calling other
        methods of the Sampler.

        Args:
          inputs: A (structure of) input tensors, it could be a nested tuple or
            a single tensor.
          **kwargs: Other kwargs for initialization. It could contain tensors
            like mask for inputs, or non tensor parameter.

        Returns:
          `(initial_finished, initial_inputs)`.
        """
        pass

    @abc.abstractmethod
    def sample(self, time, outputs, state):
        """Returns `sample_ids`."""
        pass

    @abc.abstractmethod
    def next_inputs(self, time, outputs, state, sample_ids):
        """Returns `(finished, next_inputs, next_state)`."""
        pass

    @abc.abstractproperty
    def batch_size(self):
        """Batch size of tensor returned by `sample`.

        Returns a scalar int32 tensor. The return value might not
        available before the invocation of initialize(), in this case,
        ValueError is raised.
        """
        raise NotImplementedError("batch_size has not been implemented")

    @abc.abstractproperty
    def sample_ids_shape(self):
        """Shape of tensor returned by `sample`, excluding the batch dimension.

        Returns a `TensorShape`. The return value might not available
        before the invocation of initialize().
        """
        raise NotImplementedError("sample_ids_shape has not been implemented")

    @abc.abstractproperty
    def sample_ids_dtype(self):
        """DType of tensor returned by `sample`.

        Returns a DType. The return value might not available before the
        invocation of initialize().
        """
        raise NotImplementedError("sample_ids_dtype has not been implemented")


class CustomSampler(Sampler):
    """Base abstract class that allows the user to customize sampling."""

    @typechecked
    def __init__(
        self,
        initialize_fn: Initializer,
        sample_fn: Callable,
        next_inputs_fn: Callable,
        sample_ids_shape: Optional[TensorLike] = None,
        sample_ids_dtype: types.AcceptableDTypes = None,
    ):
        """Initializer.

        Args:
          initialize_fn: callable that returns `(finished, next_inputs)` for
            the first iteration.
          sample_fn: callable that takes `(time, outputs, state)` and emits
            tensor `sample_ids`.
          next_inputs_fn: callable that takes
            `(time, outputs, state, sample_ids)` and emits
            `(finished, next_inputs, next_state)`.
          sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
            `int32`, the shape of each value in the `sample_ids` batch.
            Defaults to a scalar.
          sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to
            int32.
        """
        self._initialize_fn = initialize_fn
        self._sample_fn = sample_fn
        self._next_inputs_fn = next_inputs_fn
        self._batch_size = None
        self._sample_ids_shape = tf.TensorShape(sample_ids_shape or [])
        self._sample_ids_dtype = sample_ids_dtype or tf.int32

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return self._sample_ids_shape

    @property
    def sample_ids_dtype(self):
        return self._sample_ids_dtype

    def initialize(self, inputs, **kwargs):
        (finished, next_inputs) = self._initialize_fn(inputs, **kwargs)
        if self._batch_size is None:
            self._batch_size = tf.size(finished)
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        return self._sample_fn(time=time, outputs=outputs, state=state)

    def next_inputs(self, time, outputs, state, sample_ids):
        return self._next_inputs_fn(
            time=time, outputs=outputs, state=state, sample_ids=sample_ids
        )


class TrainingSampler(Sampler):
    """A Sampler for use during training.

    Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    @typechecked
    def __init__(self, time_major: bool = False):
        """Initializer.

        Args:
          time_major: Python bool.  Whether the tensors in `inputs` are time
            major. If `False` (default), they are assumed to be batch major.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor or `mask` is
            not a 2D boolean tensor.
        """
        self.time_major = time_major
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, inputs, sequence_length=None, mask=None):
        """Initialize the TrainSampler.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.
          mask: A boolean 2D tensor.

        Returns:
          (finished, next_inputs), a tuple of two items. The first item is a
            boolean vector to indicate whether the item in the batch has
            finished. The second item is the first slide of input data based on
            the timestep dimension (usually the second dim of the input).
        """
        self.inputs = tf.convert_to_tensor(inputs, name="inputs")
        if not self.time_major:
            inputs = tf.nest.map_structure(_transpose_batch_time, inputs)

        self._batch_size = tf.shape(tf.nest.flatten(inputs)[0])[1]

        self.input_tas = tf.nest.map_structure(_unstack_ta, inputs)
        if sequence_length is not None and mask is not None:
            raise ValueError(
                "sequence_length and mask can't be provided at the same time."
            )
        if sequence_length is not None:
            self.sequence_length = tf.convert_to_tensor(
                sequence_length, name="sequence_length"
            )
            if self.sequence_length.shape.ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be vector, but received "
                    "shape: %s" % self.sequence_length.shape
                )
        elif mask is not None:
            mask = tf.convert_to_tensor(mask)
            if mask.shape.ndims != 2:
                raise ValueError(
                    "Expected mask to a 2D tensor, but received shape: %s" % mask
                )
            if not mask.dtype.is_bool:
                raise ValueError(
                    "Expected mask to be a boolean tensor, but received "
                    "dtype: %s" % repr(mask.dtype)
                )

            axis = 1 if not self.time_major else 0
            with tf.control_dependencies(
                [_check_sequence_is_right_padded(mask, self.time_major)]
            ):
                self.sequence_length = tf.math.reduce_sum(
                    tf.cast(mask, tf.int32), axis=axis, name="sequence_length"
                )
        else:
            # As the input tensor has been converted to time major,
            # the maximum sequence length should be inferred from
            # the first dimension.
            max_seq_len = tf.shape(tf.nest.flatten(inputs)[0])[0]
            self.sequence_length = tf.fill(
                [self.batch_size], max_seq_len, name="sequence_length"
            )

        self.zero_inputs = tf.nest.map_structure(
            lambda inp: tf.zeros_like(inp[0, :]), inputs
        )

        finished = tf.equal(0, self.sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: self.zero_inputs,
            lambda: tf.nest.map_structure(lambda inp: inp.read(0), self.input_tas),
        )
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        del state
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        del sample_ids
        next_time = time + 1
        finished = next_time >= self.sequence_length
        all_finished = tf.reduce_all(finished)

        def read_from_ta(inp):
            return inp.read(next_time)

        next_inputs = tf.cond(
            all_finished,
            lambda: self.zero_inputs,
            lambda: tf.nest.map_structure(read_from_ta, self.input_tas),
        )
        return (finished, next_inputs, state)


class ScheduledEmbeddingTrainingSampler(TrainingSampler):
    """A training sampler that adds scheduled sampling.

    Returns -1s for sample_ids where no sampling took place; valid
    sample id values elsewhere.
    """

    @typechecked
    def __init__(
        self,
        sampling_probability: TensorLike,
        embedding_fn: Optional[Callable] = None,
        time_major: bool = False,
        seed: Optional[int] = None,
        scheduling_seed: Optional[TensorLike] = None,
    ):
        """Initializer.

        Args:
          sampling_probability: A `float32` 0-D or 1-D tensor: the probability
            of sampling categorically from the output ids instead of reading
            directly from the inputs.
          embedding_fn: A callable that takes a vector tensor of `ids`
            (argmax ids).
          time_major: Python bool. Whether the tensors in `inputs` are time
            major. If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          scheduling_seed: The schedule decision rule sampling seed.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        self.embedding_fn = embedding_fn
        if isinstance(sampling_probability, tf.Variable):
            self.sampling_probability = sampling_probability
        else:
            self.sampling_probability = tf.convert_to_tensor(
                sampling_probability, name="sampling_probability"
            )
        if self.sampling_probability.shape.ndims not in (0, 1):
            raise ValueError(
                "sampling_probability must be either a scalar or a vector. "
                "saw shape: %s" % (self.sampling_probability.shape)
            )
        self.seed = seed
        self.scheduling_seed = scheduling_seed
        super().__init__(time_major=time_major)

    def initialize(self, inputs, sequence_length=None, mask=None, embedding=None):
        if self.embedding_fn is None:
            if embedding is None:
                raise ValueError(
                    "embedding is required as a keyword argument for "
                    "ScheduledEmbeddingTrainingSampler"
                )
            self.embedding_fn = lambda ids: tf.nn.embedding_lookup(embedding, ids)
        return super().initialize(inputs, sequence_length=sequence_length, mask=mask)

    def sample(self, time, outputs, state):
        del state
        # Return -1s where we did not sample, and sample_ids elsewhere
        select_sample = bernoulli_sample(
            probs=self.sampling_probability,
            dtype=tf.bool,
            sample_shape=self.batch_size,
            seed=self.scheduling_seed,
        )
        return tf.where(
            select_sample,
            categorical_sample(logits=outputs, seed=self.seed),
            tf.fill([self.batch_size], -1),
        )

    def next_inputs(self, time, outputs, state, sample_ids):
        (finished, base_next_inputs, state) = super().next_inputs(
            time=time, outputs=outputs, state=state, sample_ids=sample_ids
        )

        def maybe_sample():
            """Perform scheduled sampling."""
            where_sampling = tf.cast(tf.where(sample_ids > -1), tf.int32)
            where_not_sampling = tf.cast(tf.where(sample_ids <= -1), tf.int32)
            sample_ids_sampling = tf.gather_nd(sample_ids, where_sampling)
            inputs_not_sampling = tf.gather_nd(base_next_inputs, where_not_sampling)
            sampled_next_inputs = self.embedding_fn(sample_ids_sampling)
            sampled_next_inputs = tf.cast(
                sampled_next_inputs, inputs_not_sampling.dtype
            )
            base_shape = tf.shape(base_next_inputs)
            return tf.scatter_nd(
                indices=where_sampling, updates=sampled_next_inputs, shape=base_shape
            ) + tf.scatter_nd(
                indices=where_not_sampling,
                updates=inputs_not_sampling,
                shape=base_shape,
            )

        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(all_finished, lambda: base_next_inputs, maybe_sample)
        return (finished, next_inputs, state)


class ScheduledOutputTrainingSampler(TrainingSampler):
    """A training sampler that adds scheduled sampling directly to outputs.

    Returns False for sample_ids where no sampling took place; True
    elsewhere.
    """

    @typechecked
    def __init__(
        self,
        sampling_probability: TensorLike,
        time_major: bool = False,
        seed: Optional[int] = None,
        next_inputs_fn: Optional[Callable] = None,
    ):
        """Initializer.

        Args:
          sampling_probability: A `float32` scalar tensor: the probability of
            sampling from the outputs instead of reading directly from the
            inputs.
          time_major: Python bool. Whether the tensors in `inputs` are time
            major. If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          next_inputs_fn: (Optional) callable to apply to the RNN outputs to
            create the next input when sampling. If `None` (default), the RNN
            outputs will be used as the next inputs.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        if isinstance(sampling_probability, tf.Variable):
            self.sampling_probability = sampling_probability
        else:
            self.sampling_probability = tf.convert_to_tensor(
                sampling_probability, name="sampling_probability"
            )
        if self.sampling_probability.shape.ndims not in (0, 1):
            raise ValueError(
                "sampling_probability must be either a scalar or a vector. "
                "saw shape: %s" % (self.sampling_probability.shape)
            )

        self.seed = seed
        self.next_inputs_fn = next_inputs_fn

        super().__init__(time_major=time_major)

    def initialize(
        self, inputs, sequence_length=None, mask=None, auxiliary_inputs=None
    ):
        if auxiliary_inputs is None:
            maybe_concatenated_inputs = inputs
        else:
            inputs = tf.convert_to_tensor(inputs)
            auxiliary_inputs = tf.convert_to_tensor(auxiliary_inputs)
            maybe_concatenated_inputs = tf.nest.map_structure(
                lambda x, y: tf.concat((x, y), -1), inputs, auxiliary_inputs
            )
            if not self.time_major:
                auxiliary_inputs = tf.nest.map_structure(
                    _transpose_batch_time, auxiliary_inputs
                )
        if auxiliary_inputs is not None:
            self._auxiliary_input_tas = tf.nest.map_structure(
                _unstack_ta, auxiliary_inputs
            )
        else:
            self._auxiliary_input_tas = None

        return super().initialize(
            maybe_concatenated_inputs, sequence_length=sequence_length, mask=mask
        )

    def sample(self, time, outputs, state):
        del state
        return bernoulli_sample(
            probs=self.sampling_probability,
            sample_shape=self.batch_size,
            seed=self.seed,
        )

    def next_inputs(self, time, outputs, state, sample_ids):
        (finished, base_next_inputs, state) = super().next_inputs(
            time=time, outputs=outputs, state=state, sample_ids=sample_ids
        )
        sample_ids = tf.cast(sample_ids, tf.bool)

        def maybe_sample():
            """Perform scheduled sampling."""

            def maybe_concatenate_auxiliary_inputs(outputs_, indices=None):
                """Concatenate outputs with auxiliary inputs, if they exist."""
                if self._auxiliary_input_tas is None:
                    return outputs_

                next_time = time + 1
                auxiliary_inputs = tf.nest.map_structure(
                    lambda ta: ta.read(next_time), self._auxiliary_input_tas
                )
                if indices is not None:
                    auxiliary_inputs = tf.gather_nd(auxiliary_inputs, indices)
                return tf.nest.map_structure(
                    lambda x, y: tf.concat((x, y), -1), outputs_, auxiliary_inputs
                )

            if self.next_inputs_fn is None:
                return tf.where(
                    tf.broadcast_to(
                        tf.expand_dims(sample_ids, axis=-1), base_next_inputs.shape
                    ),
                    maybe_concatenate_auxiliary_inputs(outputs),
                    base_next_inputs,
                )

            where_sampling = tf.cast(tf.where(sample_ids), tf.int32)
            where_not_sampling = tf.cast(tf.where(tf.logical_not(sample_ids)), tf.int32)
            outputs_sampling = tf.gather_nd(outputs, where_sampling)
            inputs_not_sampling = tf.gather_nd(base_next_inputs, where_not_sampling)
            sampled_next_inputs = maybe_concatenate_auxiliary_inputs(
                self.next_inputs_fn(outputs_sampling), where_sampling
            )

            base_shape = tf.shape(base_next_inputs)
            return tf.scatter_nd(
                indices=where_sampling, updates=sampled_next_inputs, shape=base_shape
            ) + tf.scatter_nd(
                indices=where_not_sampling,
                updates=inputs_not_sampling,
                shape=base_shape,
            )

        all_finished = tf.reduce_all(finished)
        no_samples = tf.logical_not(tf.reduce_any(sample_ids))
        next_inputs = tf.cond(
            tf.logical_or(all_finished, no_samples),
            lambda: base_next_inputs,
            maybe_sample,
        )
        return (finished, next_inputs, state)


class GreedyEmbeddingSampler(Sampler):
    """A sampler for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    @typechecked
    def __init__(self, embedding_fn: Optional[Callable] = None):
        """Initializer.

        Args:
          embedding_fn: A optional callable that takes a vector tensor of `ids`
            (argmax ids). The returned tensor will be passed to the decoder
            input. Default to use `tf.nn.embedding_lookup`.
        """
        self.embedding_fn = embedding_fn
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, embedding, start_tokens=None, end_token=None):
        """Initialize the GreedyEmbeddingSampler.

        Args:
          embedding: tensor that contains embedding states matrix. It will be
            used to generate generate outputs with start_tokens and end_tokens.
            The embedding will be ignored if the embedding_fn has been provided
            at __init__().
          start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
          end_token: `int32` scalar, the token that marks end of decoding.

        Returns:
          Tuple of two items: `(finished, self.start_inputs)`.
        Raises:
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
            not a scalar.
        """
        if self.embedding_fn is None:
            self.embedding_fn = lambda ids: tf.nn.embedding_lookup(embedding, ids)

        self.start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens"
        )
        self.end_token = tf.convert_to_tensor(
            end_token, dtype=tf.int32, name="end_token"
        )
        if self.start_tokens.shape.ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = tf.size(start_tokens)
        if self.end_token.shape.ndims != 0:
            raise ValueError("end_token must be a scalar")
        self.start_inputs = self.embedding_fn(self.start_tokens)

        finished = tf.tile([False], [self._batch_size])
        return (finished, self.start_inputs)

    def sample(self, time, outputs, state):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, tf.Tensor):
            raise TypeError(
                "Expected outputs to be a single Tensor, got: %s" % type(outputs)
            )
        sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self.end_token)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self.start_inputs,
            lambda: self.embedding_fn(sample_ids),
        )
        return (finished, next_inputs, state)


class SampleEmbeddingSampler(GreedyEmbeddingSampler):
    """A sampler for use during inference.

    Uses sampling (from a distribution) instead of argmax and passes the
    result through an embedding layer to get the next input.
    """

    @typechecked
    def __init__(
        self,
        embedding_fn: Optional[Callable] = None,
        softmax_temperature: Optional[TensorLike] = None,
        seed: Optional[TensorLike] = None,
    ):
        """Initializer.

        Args:
          embedding_fn: (Optional) A callable that takes a vector tensor of
            `ids` (argmax ids). The returned tensor will be passed to the
            decoder input.
          softmax_temperature: (Optional) `float32` scalar, value to divide the
            logits by before computing the softmax. Larger values (above 1.0)
            result in more random samples, while smaller values push the
            sampling distribution towards the argmax. Must be strictly greater
            than 0. Defaults to 1.0.
          seed: (Optional) The sampling seed.

        Raises:
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is
            not a scalar.
        """
        super().__init__(embedding_fn)
        self.softmax_temperature = softmax_temperature
        self.seed = seed

    def sample(self, time, outputs, state):
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, tf.Tensor):
            raise TypeError(
                "Expected outputs to be a single Tensor, got: %s" % type(outputs)
            )
        if self.softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self.softmax_temperature

        return categorical_sample(logits=logits, seed=self.seed)


class InferenceSampler(Sampler):
    """A helper to use during inference with a custom sampling function."""

    @typechecked
    def __init__(
        self,
        sample_fn: Callable,
        sample_shape: TensorLike,
        sample_dtype: types.AcceptableDTypes,
        end_fn: Callable,
        next_inputs_fn: Optional[Callable] = None,
    ):
        """Initializer.

        Args:
          sample_fn: A callable that takes `outputs` and emits tensor
            `sample_ids`.
          sample_shape: Either a list of integers, or a 1-D Tensor of type
            `int32`, the shape of the each sample in the batch returned by
            `sample_fn`.
          sample_dtype: the dtype of the sample returned by `sample_fn`.
          end_fn: A callable that takes `sample_ids` and emits a `bool` vector
            shaped `[batch_size]` indicating whether each sample is an end
            token.
          next_inputs_fn: (Optional) A callable that takes `sample_ids` and
            returns the next batch of inputs. If not provided, `sample_ids` is
            used as the next batch of inputs.
        """
        self.sample_fn = sample_fn
        self.sample_shape = tf.TensorShape(sample_shape)
        self.sample_dtype = sample_dtype
        self.end_fn = end_fn
        self.next_inputs_fn = next_inputs_fn
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return self.sample_shape

    @property
    def sample_ids_dtype(self):
        return self.sample_dtype

    def initialize(self, start_inputs):
        self.start_inputs = tf.convert_to_tensor(start_inputs, name="start_inputs")
        self._batch_size = tf.shape(start_inputs)[0]
        finished = tf.tile([False], [self._batch_size])
        return (finished, self.start_inputs)

    def sample(self, time, outputs, state):
        del time, state  # unused by sample
        return self.sample_fn(outputs)

    def next_inputs(self, time, outputs, state, sample_ids):
        del time, outputs  # unused by next_inputs
        if self.next_inputs_fn is None:
            next_inputs = sample_ids
        else:
            next_inputs = self.next_inputs_fn(sample_ids)
        finished = self.end_fn(sample_ids)
        return (finished, next_inputs, state)


# The following sample functions (_call_sampler, bernoulli_sample,
# categorical_sample) mimic TensorFlow Probability distribution semantics.
def _call_sampler(sample_n_fn, sample_shape, name=None):
    """Reshapes vector of samples."""
    with tf.name_scope(name or "call_sampler"):
        sample_shape = tf.convert_to_tensor(
            sample_shape, dtype=tf.int32, name="sample_shape"
        )
        # Ensure sample_shape is a vector (vs just a scalar).
        pad = tf.cast(tf.equal(tf.rank(sample_shape), 0), tf.int32)
        sample_shape = tf.reshape(
            sample_shape,
            tf.pad(tf.shape(sample_shape), paddings=[[pad, 0]], constant_values=1),
        )
        samples = sample_n_fn(tf.reduce_prod(sample_shape))
        batch_event_shape = tf.shape(samples)[1:]
        final_shape = tf.concat([sample_shape, batch_event_shape], 0)
        return tf.reshape(samples, final_shape)


def bernoulli_sample(
    probs=None, logits=None, dtype=tf.int32, sample_shape=(), seed=None
):
    """Samples from Bernoulli distribution."""
    if probs is None:
        probs = tf.sigmoid(logits, name="probs")
    else:
        probs = tf.convert_to_tensor(probs, name="probs")
    batch_shape_tensor = tf.shape(probs)

    def _sample_n(n):
        """Sample vector of Bernoullis."""
        new_shape = tf.concat([[n], batch_shape_tensor], 0)
        uniform = tf.random.uniform(new_shape, seed=seed, dtype=probs.dtype)
        return tf.cast(tf.less(uniform, probs), dtype)

    return _call_sampler(_sample_n, sample_shape)


def categorical_sample(logits, dtype=tf.int32, sample_shape=(), seed=None):
    """Samples from categorical distribution."""
    logits = tf.convert_to_tensor(logits, name="logits")
    event_size = tf.shape(logits)[-1]
    batch_shape_tensor = tf.shape(logits)[:-1]

    def _sample_n(n):
        """Sample vector of categoricals."""
        if logits.shape.ndims == 2:
            logits_2d = logits
        else:
            logits_2d = tf.reshape(logits, [-1, event_size])
        sample_dtype = tf.int64 if logits.dtype.size > 4 else tf.int32
        draws = tf.random.categorical(logits_2d, n, dtype=sample_dtype, seed=seed)
        draws = tf.reshape(tf.transpose(draws), tf.concat([[n], batch_shape_tensor], 0))
        return tf.cast(draws, dtype)

    return _call_sampler(_sample_n, sample_shape)


def _unstack_ta(inp):
    return tf.TensorArray(
        dtype=inp.dtype, size=tf.shape(inp)[0], element_shape=inp.shape[1:]
    ).unstack(inp)


def _check_sequence_is_right_padded(mask, time_major):
    """Returns an Assert operation checking that if the mask tensor is right
    padded."""
    if time_major:
        mask = tf.transpose(mask)
    sequence_length = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    max_seq_length = tf.shape(mask)[1]
    right_padded_mask = tf.sequence_mask(
        sequence_length, maxlen=max_seq_length, dtype=tf.bool
    )
    all_equal = tf.math.equal(mask, right_padded_mask)

    condition = tf.math.reduce_all(all_equal)
    error_message = "The input sequence should be right padded."

    return tf.Assert(condition, [error_message])
