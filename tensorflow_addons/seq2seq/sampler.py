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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow_addons.seq2seq import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

__all__ = [
    "Sampler",
    "TrainingSampler",
    "GreedyEmbeddingSampler",
    "SampleEmbeddingSampler",
    "CustomSampler",
    "ScheduledEmbeddingTrainingSampler",
    "ScheduledOutputTrainingSampler",
    "InferenceSampler",
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


@six.add_metaclass(abc.ABCMeta)
class Sampler(object):
    """Interface for implementing sampling in seq2seq decoders.

    Sampler instances are used by `BasicDecoder`. The normal usage of a sampler
    is like below:
    sampler = Sampler(init_args)
    (initial_finished, initial_inputs) = sampler.initialize(input_tensors)
    for time_step in range(time):
      cell_output, cell_state = cell.call(cell_input, previous_state)
      sample_ids = sampler.sample(time_step, cell_output, cell_state)
      (finished, next_inputs, next_state) = sampler.next_inputs(
          time_step,cell_output, cell_state)

    Note that all the tensor input should not be feed to Sampler as __init__()
    parameters, instead, they should be feed by decoders via initialize().
    """

    @abc.abstractmethod
    def initialize(self, inputs, **kwargs):
        """initialize the sampler with the input tensors.

        This method suppose to be only invoke once before the calling other
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

    def __init__(self,
                 initialize_fn,
                 sample_fn,
                 next_inputs_fn,
                 sample_ids_shape=None,
                 sample_ids_dtype=None):
        """Initializer.

        Args:
          initialize_fn: callable that returns `(finished, next_inputs)` for the
            first iteration.
          sample_fn: callable that takes `(time, outputs, state)` and emits
            tensor `sample_ids`.
          next_inputs_fn: callable that takes
            `(time, outputs, state, sample_ids)` and emits
            `(finished, next_inputs, next_state)`.
          sample_ids_shape: Either a list of integers, or a 1-D Tensor of type
            `int32`, the shape of each value in the `sample_ids` batch. Defaults
            to a scalar.
          sample_ids_dtype: The dtype of the `sample_ids` tensor. Defaults to
            int32.
        """
        self._initialize_fn = initialize_fn
        self._sample_fn = sample_fn
        self._next_inputs_fn = next_inputs_fn
        self._batch_size = None
        self._sample_ids_shape = tensor_shape.TensorShape(sample_ids_shape
                                                          or [])
        self._sample_ids_dtype = sample_ids_dtype or dtypes.int32

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError(
                "batch_size accessed before initialize was called")
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
            self._batch_size = array_ops.size(finished)
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        return self._sample_fn(time=time, outputs=outputs, state=state)

    def next_inputs(self, time, outputs, state, sample_ids):
        return self._next_inputs_fn(
            time=time, outputs=outputs, state=state, sample_ids=sample_ids)


class TrainingSampler(Sampler):
    """A Sampler for use during training.

    Only reads inputs.

    Returned sample_ids are the argmax of the RNN output logits.
    """

    def __init__(self, time_major=False):
        """Initializer.

        Args:
          time_major: Python bool.  Whether the tensors in `inputs` are time
            major. If `False` (default), they are assumed to be batch major.

        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """
        self.time_major = time_major
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError(
                "batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

    def initialize(self, inputs, sequence_length=None):
        """Initialize the TrainSampler.

        Args:
          inputs: A (structure of) input tensors.
          sequence_length: An int32 vector tensor.

        Returns:
          (finished, next_inputs), a tuple of two items. The first item is a
            boolean vector to indicate whether the item in the batch has
            finished. The second item is the first slide of input data based on
            the timestep dimension (usually the second dim of the input).
        """
        self.inputs = ops.convert_to_tensor(inputs, name="inputs")
        if not self.time_major:
            inputs = nest.map_structure(_transpose_batch_time, inputs)

        self.input_tas = nest.map_structure(_unstack_ta, inputs)
        if sequence_length is None:
            raise ValueError("sequence_length is required for TrainingSampler")
        self.sequence_length = ops.convert_to_tensor(
            sequence_length, name="sequence_length")
        if self.sequence_length.get_shape().ndims != 1:
            raise ValueError(
                "Expected sequence_length to be vector, but received shape: %s"
                % self._sequence_length.get_shape())

        self.zero_inputs = nest.map_structure(
            lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

        self._batch_size = array_ops.size(self.sequence_length)

        finished = math_ops.equal(0, self.sequence_length)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished, lambda: self.zero_inputs, lambda: nest.map_structure(
                lambda inp: inp.read(0), self.input_tas))
        return (finished, next_inputs)

    def sample(self, time, outputs, state):
        del state
        sample_ids = math_ops.cast(
            math_ops.argmax(outputs, axis=-1), dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        del sample_ids
        next_time = time + 1
        finished = (next_time >= self.sequence_length)
        all_finished = math_ops.reduce_all(finished)

        def read_from_ta(inp):
            return inp.read(next_time)

        next_inputs = control_flow_ops.cond(
            all_finished, lambda: self.zero_inputs, lambda: nest.map_structure(
                read_from_ta, self.input_tas))
        return (finished, next_inputs, state)


class ScheduledEmbeddingTrainingSampler(TrainingSampler):
    """A training sampler that adds scheduled sampling.

    Returns -1s for sample_ids where no sampling took place; valid
    sample id values elsewhere.
    """

    def __init__(self,
                 sampling_probability,
                 embedding_fn=None,
                 time_major=False,
                 seed=None,
                 scheduling_seed=None):
        """Initializer.

        Args:
          sampling_probability: A `float32` 0-D or 1-D tensor: the probability
            of sampling categorically from the output ids instead of reading
            directly from the inputs.
          embedding_fn: A callable that takes a vector tensor of `ids`
            (argmax ids), or the `params` argument for `embedding_lookup`.
          time_major: Python bool. Whether the tensors in `inputs` are time
            major. If `False` (default), they are assumed to be batch major.
          seed: The sampling seed.
          scheduling_seed: The schedule decision rule sampling seed.

        Raises:
          ValueError: if `sampling_probability` is not a scalar or vector.
        """
        if callable(embedding_fn) or embedding_fn is None:
            self.embedding_fn = embedding_fn
        else:
            raise ValueError("embedding_fn is expected to be callable, got %s"
                             % type(embedding_fn))
        self.sampling_probability = ops.convert_to_tensor(
            sampling_probability, name="sampling_probability")
        if self.sampling_probability.get_shape().ndims not in (0, 1):
            raise ValueError(
                "sampling_probability must be either a scalar or a vector. "
                "saw shape: %s" % (self.sampling_probability.get_shape()))
        self.seed = seed
        self.scheduling_seed = scheduling_seed
        super(ScheduledEmbeddingTrainingSampler,
              self).__init__(time_major=time_major)

    def initialize(self, inputs, sequence_length=None, embedding=None):
        if self.embedding_fn is None:
            if embedding is None:
                raise ValueError(
                    "embedding is required as a keyword argument for "
                    "ScheduledEmbeddingTrainingSampler")
            self.embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))
        return super(ScheduledEmbeddingTrainingSampler, self).initialize(
            inputs, sequence_length=sequence_length)

    def sample(self, time, outputs, state):
        del state
        # Return -1s where we did not sample, and sample_ids elsewhere
        select_sample = bernoulli_sample(
            probs=self.sampling_probability,
            dtype=dtypes.bool,
            sample_shape=self.batch_size,
            seed=self.scheduling_seed)
        return array_ops.where(
            select_sample, categorical_sample(logits=outputs, seed=self.seed),
            gen_array_ops.fill([self.batch_size], -1))

    def next_inputs(self, time, outputs, state, sample_ids):
        (finished, base_next_inputs,
         state) = (super(ScheduledEmbeddingTrainingSampler, self).next_inputs(
             time=time, outputs=outputs, state=state, sample_ids=sample_ids))

        def maybe_sample():
            """Perform scheduled sampling."""
            where_sampling = math_ops.cast(
                array_ops.where(sample_ids > -1), dtypes.int32)
            where_not_sampling = math_ops.cast(
                array_ops.where(sample_ids <= -1), dtypes.int32)
            sample_ids_sampling = array_ops.gather_nd(sample_ids,
                                                      where_sampling)
            inputs_not_sampling = array_ops.gather_nd(base_next_inputs,
                                                      where_not_sampling)
            sampled_next_inputs = self.embedding_fn(sample_ids_sampling)
            base_shape = array_ops.shape(base_next_inputs)
            return (array_ops.scatter_nd(
                indices=where_sampling,
                updates=sampled_next_inputs,
                shape=base_shape) + array_ops.scatter_nd(
                    indices=where_not_sampling,
                    updates=inputs_not_sampling,
                    shape=base_shape))

        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished, lambda: base_next_inputs, maybe_sample)
        return (finished, next_inputs, state)


class ScheduledOutputTrainingSampler(TrainingSampler):
    """A training sampler that adds scheduled sampling directly to outputs.

    Returns False for sample_ids where no sampling took place; True
    elsewhere.
    """

    def __init__(self,
                 sampling_probability,
                 time_major=False,
                 seed=None,
                 next_inputs_fn=None):
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
        self.sampling_probability = ops.convert_to_tensor(
            sampling_probability, name="sampling_probability")
        if self.sampling_probability.get_shape().ndims not in (0, 1):
            raise ValueError(
                "sampling_probability must be either a scalar or a vector. "
                "saw shape: %s" % (self._sampling_probability.get_shape()))

        self.seed = seed
        self.next_inputs_fn = next_inputs_fn

        super(ScheduledOutputTrainingSampler,
              self).__init__(time_major=time_major)

    def initialize(self, inputs, sequence_length=None, auxiliary_inputs=None):
        if auxiliary_inputs is None:
            maybe_concatenated_inputs = inputs
        else:
            inputs = ops.convert_to_tensor(inputs)
            auxiliary_inputs = ops.convert_to_tensor(auxiliary_inputs)
            maybe_concatenated_inputs = nest.map_structure(
                lambda x, y: array_ops.concat((x, y), -1), inputs,
                auxiliary_inputs)
            if not self.time_major:
                auxiliary_inputs = nest.map_structure(_transpose_batch_time,
                                                      auxiliary_inputs)
        if auxiliary_inputs is not None:
            self._auxiliary_input_tas = nest.map_structure(
                _unstack_ta, auxiliary_inputs)
        else:
            self._auxiliary_input_tas = None

        return super(ScheduledOutputTrainingSampler, self).initialize(
            maybe_concatenated_inputs, sequence_length=sequence_length)

    def sample(self, time, outputs, state):
        del state
        return bernoulli_sample(
            probs=self.sampling_probability,
            sample_shape=self.batch_size,
            seed=self.seed)

    def next_inputs(self, time, outputs, state, sample_ids):
        (finished, base_next_inputs,
         state) = (super(ScheduledOutputTrainingSampler, self).next_inputs(
             time=time, outputs=outputs, state=state, sample_ids=sample_ids))
        sample_ids = math_ops.cast(sample_ids, dtypes.bool)

        def maybe_sample():
            """Perform scheduled sampling."""

            def maybe_concatenate_auxiliary_inputs(outputs_, indices=None):
                """Concatenate outputs with auxiliary inputs, if they exist."""
                if self._auxiliary_input_tas is None:
                    return outputs_

                next_time = time + 1
                auxiliary_inputs = nest.map_structure(
                    lambda ta: ta.read(next_time), self._auxiliary_input_tas)
                if indices is not None:
                    auxiliary_inputs = array_ops.gather_nd(
                        auxiliary_inputs, indices)
                return nest.map_structure(
                    lambda x, y: array_ops.concat((x, y), -1), outputs_,
                    auxiliary_inputs)

            if self.next_inputs_fn is None:
                return array_ops.where(
                    sample_ids, maybe_concatenate_auxiliary_inputs(outputs),
                    base_next_inputs)

            where_sampling = math_ops.cast(
                array_ops.where(sample_ids), dtypes.int32)
            where_not_sampling = math_ops.cast(
                array_ops.where(math_ops.logical_not(sample_ids)),
                dtypes.int32)
            outputs_sampling = array_ops.gather_nd(outputs, where_sampling)
            inputs_not_sampling = array_ops.gather_nd(base_next_inputs,
                                                      where_not_sampling)
            sampled_next_inputs = maybe_concatenate_auxiliary_inputs(
                self.next_inputs_fn(outputs_sampling), where_sampling)

            base_shape = array_ops.shape(base_next_inputs)
            return (array_ops.scatter_nd(
                indices=where_sampling,
                updates=sampled_next_inputs,
                shape=base_shape) + array_ops.scatter_nd(
                    indices=where_not_sampling,
                    updates=inputs_not_sampling,
                    shape=base_shape))

        all_finished = math_ops.reduce_all(finished)
        no_samples = math_ops.logical_not(math_ops.reduce_any(sample_ids))
        next_inputs = control_flow_ops.cond(
            math_ops.logical_or(all_finished,
                                no_samples), lambda: base_next_inputs,
            maybe_sample)
        return (finished, next_inputs, state)


class GreedyEmbeddingSampler(Sampler):
    """A sampler for use during inference.

    Uses the argmax of the output (treated as logits) and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding_fn=None):
        """Initializer.

        Args:
          embedding_fn: A optional callable that takes a vector tensor of `ids`
            (argmax ids), or the `params` argument for `embedding_lookup`. The
            returned tensor will be passed to the decoder input. Default to use
            `embedding_ops.embedding_lookup`.
        """
        if embedding_fn is None or callable(embedding_fn):
            self.embedding_fn = embedding_fn
        else:
            raise ValueError(
                "embedding_fn is expected to be a callable, got %s" %
                type(embedding_fn))
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError(
                "batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32

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
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not
            a scalar.
        """
        if self.embedding_fn is None:
            self.embedding_fn = (
                lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self.start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        self.end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self.start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self.end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self.start_inputs = self.embedding_fn(self.start_tokens)

        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self.start_inputs)

    def sample(self, time, outputs, state):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        sample_ids = math_ops.argmax(
            outputs, axis=-1, output_type=dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self.end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self.start_inputs,
            lambda: self.embedding_fn(sample_ids))
        return (finished, next_inputs, state)


class SampleEmbeddingSampler(GreedyEmbeddingSampler):
    """A sampler for use during inference.

    Uses sampling (from a distribution) instead of argmax and passes the
    result through an embedding layer to get the next input.
    """

    def __init__(self, embedding_fn=None, softmax_temperature=None, seed=None):
        """Initializer.

        Args:
          embedding_fn: (Optional) A callable that takes a vector tensor of
            `ids` (argmax ids), or the `params` argument for `embedding_lookup`.
            The returned tensor will be passed to the decoder input.
          softmax_temperature: (Optional) `float32` scalar, value to divide the
            logits by before computing the softmax. Larger values (above 1.0)
            result in more random samples, while smaller values push the
            sampling distribution towards the argmax. Must be strictly greater
            than 0. Defaults to 1.0.
          seed: (Optional) The sampling seed.

        Raises:
          ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not
            a scalar.
        """
        super(SampleEmbeddingSampler, self).__init__(embedding_fn)
        self.softmax_temperature = softmax_temperature
        self.seed = seed

    def sample(self, time, outputs, state):
        """sample for SampleEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, we sample instead of argmax (greedy).
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        if self.softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self.softmax_temperature

        return categorical_sample(logits=logits, seed=self.seed)


class InferenceSampler(Sampler):
    """A helper to use during inference with a custom sampling function."""

    def __init__(self,
                 sample_fn,
                 sample_shape,
                 sample_dtype,
                 end_fn,
                 next_inputs_fn=None):
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
        self.sample_shape = tensor_shape.TensorShape(sample_shape)
        self.sample_dtype = sample_dtype
        self.end_fn = end_fn
        self.next_inputs_fn = next_inputs_fn
        self._batch_size = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError(
                "batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return self.sample_shape

    @property
    def sample_ids_dtype(self):
        return self.sample_dtype

    def initialize(self, start_inputs):
        self.start_inputs = ops.convert_to_tensor(
            start_inputs, name="start_inputs")
        self._batch_size = array_ops.shape(start_inputs)[0]
        finished = array_ops.tile([False], [self._batch_size])
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
    with ops.name_scope(name, "call_sampler", values=[sample_shape]):
        sample_shape = ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32, name="sample_shape")
        # Ensure sample_shape is a vector (vs just a scalar).
        pad = math_ops.cast(
            math_ops.equal(array_ops.rank(sample_shape), 0), dtypes.int32)
        sample_shape = array_ops.reshape(
            sample_shape,
            array_ops.pad(
                array_ops.shape(sample_shape),
                paddings=[[pad, 0]],
                constant_values=1))
        samples = sample_n_fn(math_ops.reduce_prod(sample_shape))
        batch_event_shape = array_ops.shape(samples)[1:]
        final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
        return array_ops.reshape(samples, final_shape)


def bernoulli_sample(probs=None,
                     logits=None,
                     dtype=dtypes.int32,
                     sample_shape=(),
                     seed=None):
    """Samples from Bernoulli distribution."""
    if probs is None:
        probs = math_ops.sigmoid(logits, name="probs")
    else:
        probs = ops.convert_to_tensor(probs, name="probs")
    batch_shape_tensor = array_ops.shape(probs)

    def _sample_n(n):
        """Sample vector of Bernoullis."""
        new_shape = array_ops.concat([[n], batch_shape_tensor], 0)
        uniform = random_ops.random_uniform(
            new_shape, seed=seed, dtype=probs.dtype)
        return math_ops.cast(math_ops.less(uniform, probs), dtype)

    return _call_sampler(_sample_n, sample_shape)


def categorical_sample(logits, dtype=dtypes.int32, sample_shape=(), seed=None):
    """Samples from categorical distribution."""
    logits = ops.convert_to_tensor(logits, name="logits")
    event_size = array_ops.shape(logits)[-1]
    batch_shape_tensor = array_ops.shape(logits)[:-1]

    def _sample_n(n):
        """Sample vector of categoricals."""
        if logits.shape.ndims == 2:
            logits_2d = logits
        else:
            logits_2d = array_ops.reshape(logits, [-1, event_size])
        sample_dtype = dtypes.int64 if logits.dtype.size > 4 else dtypes.int32
        draws = random_ops.multinomial(
            logits_2d, n, seed=seed, output_dtype=sample_dtype)
        draws = array_ops.reshape(
            array_ops.transpose(draws),
            array_ops.concat([[n], batch_shape_tensor], 0))
        return math_ops.cast(draws, dtype)

    return _call_sampler(_sample_n, sample_shape)


def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
        dtype=inp.dtype,
        size=array_ops.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)
