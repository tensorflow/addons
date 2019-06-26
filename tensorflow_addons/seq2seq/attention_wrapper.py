# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A powerful dynamic attention wrapper object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np

import tensorflow as tf

# TODO: Find public API alternatives to these
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops import rnn_cell_impl

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class AttentionMechanism(object):
    @property
    def alignments_size(self):
        raise NotImplementedError

    @property
    def state_size(self):
        raise NotImplementedError


class _BaseAttentionMechanism(AttentionMechanism, tf.keras.layers.Layer):
    """A base AttentionMechanism class providing common functionality.

    Common functionality includes:
      1. Storing the query and memory layers.
      2. Preprocessing and storing the memory.

    Note that this layer takes memory as its init parameter, which is an
    anti-pattern of Keras API, we have to keep the memory as init parameter for
    performance and dependency reason. Under the hood, during `__init__()`, it
    will invoke `base_layer.__call__(memory, setup_memory=True)`. This will let
    keras to keep track of the memory tensor as the input of this layer. Once
    the `__init__()` is done, then user can query the attention by
    `score = att_obj([query, state])`, and use it as a normal keras layer.

    Special attention is needed when adding using this class as the base layer
    for new attention:
      1. Build() could be invoked at least twice. So please make sure weights
         are not duplicated.
      2. Layer.get_weights() might return different set of weights if the
         instance has `query_layer`. The query_layer weights is not initialized
         until the memory is configured.

    Also note that this layer does not work with Keras model when
    `model.compile(run_eagerly=True)` due to the fact that this layer is
    stateful. The support for that will be added in a future version.
    """

    def __init__(self,
                 memory,
                 probability_fn,
                 query_layer=None,
                 memory_layer=None,
                 memory_sequence_length=None,
                 **kwargs):
        """Construct base AttentionMechanism class.

        Args:
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          probability_fn: A `callable`. Converts the score and previous
            alignments to probabilities. Its signature should be:
            `probabilities = probability_fn(score, state)`.
          query_layer:  (optional): Instance of `tf.keras.Layer`.  The layer's
            depth must match the depth of `memory_layer`.  If `query_layer` is
            not provided, the shape of `query` must match that of
            `memory_layer`.
          memory_layer: (optional): Instance of `tf.keras.Layer`. The layer's
            depth must match the depth of `query_layer`.
            If `memory_layer` is not provided, the shape of `memory` must match
            that of `query_layer`.
          memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        if (query_layer is not None
                and not isinstance(query_layer, tf.keras.layers.Layer)):
            raise TypeError(
                "query_layer is not a Layer: %s" % type(query_layer).__name__)
        if (memory_layer is not None
                and not isinstance(memory_layer, tf.keras.layers.Layer)):
            raise TypeError("memory_layer is not a Layer: %s" %
                            type(memory_layer).__name__)
        self.query_layer = query_layer
        self.memory_layer = memory_layer
        if self.memory_layer is not None and "dtype" not in kwargs:
            kwargs["dtype"] = self.memory_layer.dtype
        super(_BaseAttentionMechanism, self).__init__(**kwargs)
        if not callable(probability_fn):
            raise TypeError("probability_fn must be callable, saw type: %s" %
                            type(probability_fn).__name__)
        self.probability_fn = probability_fn

        self.keys = None
        self.values = None
        self.batch_size = None
        self._memory_initialized = False
        self._check_inner_dims_defined = True
        self.supports_masking = True
        self.score_mask_value = tf.as_dtype(self.dtype).as_numpy_dtype(-np.inf)

        if memory is not None:
            # Setup the memory by self.__call__() with memory and
            # memory_seq_length. This will make the attention follow the keras
            # convention which takes all the tensor inputs via __call__().
            if memory_sequence_length is None:
                inputs = memory
            else:
                inputs = [memory, memory_sequence_length]

            self.values = super(_BaseAttentionMechanism, self).__call__(
                inputs, setup_memory=True)

    def build(self, input_shape):
        if not self._memory_initialized:
            # This is for setting up the memory, which contains memory and
            # optional memory_sequence_length. Build the memory_layer with
            # memory shape.
            if self.memory_layer is not None and not self.memory_layer.built:
                if isinstance(input_shape, list):
                    self.memory_layer.build(input_shape[0])
                else:
                    self.memory_layer.build(input_shape)
        else:
            # The input_shape should be query.shape and state.shape. Use the
            # query to init the query layer.
            if self.query_layer is not None and not self.query_layer.built:
                self.query_layer.build(input_shape[0])

    def __call__(self, inputs, **kwargs):
        """Preprocess the inputs before calling `base_layer.__call__()`.

        Note that there are situation here, one for setup memory, and one with
        actual query and state.
        1. When the memory has not been configured, we just pass all the param
           to base_layer.__call__(), which will then invoke self.call() with
           proper inputs, which allows this class to setup memory.
        2. When the memory has already been setup, the input should contain
           query and state, and optionally processed memory. If the processed
           memory is not included in the input, we will have to append it to
           the inputs and give it to the base_layer.__call__(). The processed
           memory is the output of first invocation of self.__call__(). If we
           don't add it here, then from keras perspective, the graph is
           disconnected since the output from previous call is never used.

        Args:
          inputs: the inputs tensors.
          **kwargs: dict, other keyeword arguments for the `__call__()`
        """
        if self._memory_initialized:
            if len(inputs) not in (2, 3):
                raise ValueError(
                    "Expect the inputs to have 2 or 3 tensors, got %d" %
                    len(inputs))
            if len(inputs) == 2:
                # We append the calculated memory here so that the graph will be
                # connected.
                inputs.append(self.values)
        return super(_BaseAttentionMechanism, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, setup_memory=False, **kwargs):
        """Setup the memory or query the attention.

        There are two case here, one for setup memory, and the second is query
        the attention score. `setup_memory` is the flag to indicate which mode
        it is. The input list will be treated differently based on that flag.

        Args:
          inputs: a list of tensor that could either be `query` and `state`, or
            `memory` and `memory_sequence_length`.
            `query` is the tensor of dtype matching `memory` and shape
            `[batch_size, query_depth]`.
            `state` is the tensor of dtype matching `memory` and shape
            `[batch_size, alignments_size]`. (`alignments_size` is memory's
            `max_time`).
            `memory` is the memory to query; usually the output of an RNN
            encoder. The tensor should be shaped `[batch_size, max_time, ...]`.
            `memory_sequence_length` (optional) is the sequence lengths for the
             batch entries in memory. If provided, the memory tensor rows are
            masked with zeros for values past the respective sequence lengths.
          mask: optional bool tensor with shape `[batch, max_time]` for the
            mask of memory. If it is not None, the corresponding item of the
            memory should be filtered out during calculation.
          setup_memory: boolean, whether the input is for setting up memory, or
            query attention.
          **kwargs: Dict, other keyword arguments for the call method.
        Returns:
          Either processed memory or attention score, based on `setup_memory`.
        """
        if setup_memory:
            if isinstance(inputs, list):
                if len(inputs) not in (1, 2):
                    raise ValueError(
                        "Expect inputs to have 1 or 2 tensors, got %d" %
                        len(inputs))
                memory = inputs[0]
                memory_sequence_length = inputs[1] if len(
                    inputs) == 2 else None
                memory_mask = mask
            else:
                memory, memory_sequence_length = inputs, None
                memory_mask = mask
            self._setup_memory(memory, memory_sequence_length, memory_mask)
            # We force the self.built to false here since only memory is,
            # initialized but the real query/state has not been call() yet. The
            # layer should be build and call again.
            self.built = False
            # Return the processed memory in order to create the Keras
            # connectivity data for it.
            return self.values
        else:
            if not self._memory_initialized:
                raise ValueError(
                    "Cannot query the attention before the setup of "
                    "memory")
            if len(inputs) not in (2, 3):
                raise ValueError(
                    "Expect the inputs to have query, state, and optional "
                    "processed memory, got %d items" % len(inputs))
            # Ignore the rest of the inputs and only care about the query and
            # state
            query, state = inputs[0], inputs[1]
            return self._calculate_attention(query, state)

    def _setup_memory(self,
                      memory,
                      memory_sequence_length=None,
                      memory_mask=None):
        """Pre-process the memory before actually query the memory.

        This should only be called once at the first invocation of call().

        Args:
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length (optional): Sequence lengths for the batch
            entries in memory. If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          memory_mask: (Optional) The boolean tensor with shape `[batch_size,
            max_time]`. For any value equal to False, the corresponding value
            in memory should be ignored.
        """
        if self._memory_initialized:
            raise ValueError(
                "The memory for the attention has already been setup.")
        if memory_sequence_length is not None and memory_mask is not None:
            raise ValueError(
                "memory_sequence_length and memory_mask cannot be "
                "used at same time for attention.")
        with tf.name_scope(self.name or "BaseAttentionMechanismInit"):
            self.values = _prepare_memory(
                memory,
                memory_sequence_length=memory_sequence_length,
                memory_mask=memory_mask,
                check_inner_dims_defined=self._check_inner_dims_defined)
            # Mark the value as check since the memory and memory mask might not
            # passed from __call__(), which does not have proper keras metadata.
            # TODO(omalleyt12): Remove this hack once the mask the has proper
            # keras history.
            base_layer_utils.mark_checked(self.values)
            if self.memory_layer is not None:
                self.keys = self.memory_layer(self.values)
            else:
                self.keys = self.values
            self.batch_size = (tf.compat.dimension_value(self.keys.shape[0])
                               or tf.shape(self.keys)[0])
            self._alignments_size = (tf.compat.dimension_value(
                self.keys.shape[1]) or tf.shape(self.keys)[1])
            if memory_mask is not None or memory_sequence_length is not None:
                unwrapped_probability_fn = self.probability_fn

                def _mask_probability_fn(score, prev):
                    return unwrapped_probability_fn(
                        _maybe_mask_score(
                            score,
                            memory_mask=memory_mask,
                            memory_sequence_length=memory_sequence_length,
                            score_mask_value=self.score_mask_value), prev)

                self.probability_fn = _mask_probability_fn
        self._memory_initialized = True

    def _calculate_attention(self, query, state):
        raise NotImplementedError(
            "_calculate_attention need to be implemented by subclasses.")

    def compute_mask(self, inputs, mask=None):
        # There real input of the attention is query and state, and the memory
        # layer mask shouldn't be pass down. Returning None for all output mask
        # here.
        return None, None

    def get_config(self):
        config = {}
        # Since the probability_fn is likely to be a wrapped function, the child
        # class should preserve the original function and how its wrapped.

        if self.query_layer is not None:
            config["query_layer"] = {
                "class_name": self.query_layer.__class__.__name__,
                "config": self.query_layer.get_config(),
            }
        if self.memory_layer is not None:
            config["memory_layer"] = {
                "class_name": self.memory_layer.__class__.__name__,
                "config": self.memory_layer.get_config(),
            }
        # memory is a required init parameter and its a tensor. It cannot be
        # serialized to config, so we put a placeholder for it.
        config["memory"] = None
        base_config = super(_BaseAttentionMechanism, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _process_probability_fn(self, func_name):
        """Helper method to retrieve the probably function by string input."""
        valid_probability_fns = {
            "softmax": tf.nn.softmax,
            "hardmax": hardmax,
        }
        if func_name not in valid_probability_fns.keys():
            raise ValueError("Invalid probability function: %s, options are %s"
                             % (func_name, valid_probability_fns.keys()))
        return valid_probability_fns[func_name]

    @classmethod
    def deserialize_inner_layer_from_config(cls, config, custom_objects):
        """Helper method that reconstruct the query and memory from the config.

        In the get_config() method, the query and memory layer configs are
        serialized into dict for persistence, this method perform the reverse
        action to reconstruct the layer from the config.

        Args:
          config: dict, the configs that will be used to reconstruct the
            object.
          custom_objects: dict mapping class names (or function names) of
            custom (non-Keras) objects to class/functions.
        Returns:
          config: dict, the config with layer instance created, which is ready
            to be used as init parameters.
        """
        # Reconstruct the query and memory layer for parent class.
        # Instead of updating the input, create a copy and use that.
        config = config.copy()
        query_layer_config = config.pop("query_layer", None)
        if query_layer_config:
            query_layer = tf.keras.layers.deserialize(
                query_layer_config, custom_objects=custom_objects)
            config["query_layer"] = query_layer
        memory_layer_config = config.pop("memory_layer", None)
        if memory_layer_config:
            memory_layer = tf.keras.layers.deserialize(
                memory_layer_config, custom_objects=custom_objects)
            config["memory_layer"] = memory_layer
        return config

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self._alignments_size

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `AttentionWrapper`
        class.

        This is important for AttentionMechanisms that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).

        The default behavior is to return a tensor of all zeros.

        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.

        Returns:
          A `dtype` tensor shaped `[batch_size, alignments_size]`
          (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        """Creates the initial state values for the `AttentionWrapper` class.

        This is important for AttentionMechanisms that use the previous
        alignment to calculate the alignment at the next time step
        (e.g. monotonic attention).

        The default behavior is to return the same output as
        initial_alignments.

        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.

        Returns:
          A structure of all-zero tensors with shapes as described by
          `state_size`.
        """
        return self.initial_alignments(batch_size, dtype)


def _luong_score(query, keys, scale):
    """Implements Luong-style (multiplicative) scoring function.

    This attention has two forms.  The first is standard Luong attention,
    as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, call this function with `scale=True`.

    Args:
      query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      scale: the optional tensor to scale the attention score.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys. "
            "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
            "Perhaps you need to set num_units to the keys' dimension (%s)?" %
            (query, depth, keys, key_units, key_units))

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = tf.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = tf.matmul(query, keys, transpose_b=True)
    score = tf.squeeze(score, [1])

    if scale is not None:
        score = scale * score
    return score


class LuongAttention(_BaseAttentionMechanism):
    """Implements Luong-style (multiplicative) attention scoring.

    This attention has two forms.  The first is standard Luong attention,
    as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    [Effective Approaches to Attention-based Neural Machine Translation.
    EMNLP 2015.](https://arxiv.org/abs/1508.04025)

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, construct the object with parameter
    `scale=True`.
    """

    def __init__(self,
                 units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn="softmax",
                 dtype=None,
                 name="LuongAttention",
                 **kwargs):
        """Construct the AttentionMechanism mechanism.

        Args:
          units: The depth of the attention mechanism.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          scale: Python boolean. Whether to scale the energy term.
          probability_fn: (optional) string, the name of function to convert
            the attention score to probabilities. The default is `softmax`
            which is `tf.nn.softmax`. Other options is `hardmax`, which is
            hardmax() within this module. Any other value will result
            intovalidation error. Default to use `softmax`.
          dtype: The data type for the memory layer of the attention mechanism.
          name: Name to use when creating ops.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        # For LuongAttention, we only transform the memory layer; thus
        # num_units **must** match expected the query depth.
        self.probability_fn_name = probability_fn
        probability_fn = self._process_probability_fn(self.probability_fn_name)
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        if dtype is None:
            dtype = tf.float32
        memory_layer = kwargs.pop("memory_layer", None)
        if not memory_layer:
            memory_layer = tf.keras.layers.Dense(
                units, name="memory_layer", use_bias=False, dtype=dtype)
        self.units = units
        self.scale = scale
        self.scale_weight = None
        super(LuongAttention, self).__init__(
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            query_layer=None,
            memory_layer=memory_layer,
            probability_fn=wrapped_probability_fn,
            name=name,
            dtype=dtype,
            **kwargs)

    def build(self, input_shape):
        super(LuongAttention, self).build(input_shape)
        if self.scale and self.scale_weight is None:
            self.scale_weight = self.add_weight(
                "attention_g", initializer=tf.ones_initializer, shape=())
        self.built = True

    def _calculate_attention(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
          next_state: Same as the alignments.
        """
        score = _luong_score(query, self.keys, self.scale_weight)
        alignments = self.probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

    def get_config(self):
        config = {
            "units": self.units,
            "scale": self.scale,
            "probability_fn": self.probability_fn_name,
        }
        base_config = super(LuongAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = _BaseAttentionMechanism.deserialize_inner_layer_from_config(
            config, custom_objects=custom_objects)
        return cls(**config)


def _bahdanau_score(processed_query,
                    keys,
                    attention_v,
                    attention_g=None,
                    attention_b=None):
    """Implements Bahdanau-style (additive) scoring function.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, set please pass in attention_g and attention_b.

    Args:
      processed_query: Tensor, shape `[batch_size, num_units]` to compare to
        keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      attention_v: Tensor, shape `[num_units]`.
      attention_g: Optional scalar tensor for normalization.
      attention_b: Optional tensor with shape `[num_units]` for normalization.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    if attention_g is not None and attention_b is not None:
        normed_v = attention_g * attention_v * tf.math.rsqrt(
            tf.reduce_sum(tf.square(attention_v)))
        return tf.reduce_sum(
            normed_v * tf.tanh(keys + processed_query + attention_b), [2])
    else:
        return tf.reduce_sum(attention_v * tf.tanh(keys + processed_query),
                             [2])


class BahdanauAttention(_BaseAttentionMechanism):
    """Implements Bahdanau-style (additive) attention.

    This attention has two forms.  The first is Bahdanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn="softmax",
                 kernel_initializer="glorot_uniform",
                 dtype=None,
                 name="BahdanauAttention",
                 **kwargs):
        """Construct the Attention mechanism.

        Args:
          units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          normalize: Python boolean.  Whether to normalize the energy term.
          probability_fn: (optional) string, the name of function to convert
            the attention score to probabilities. The default is `softmax`
            which is `tf.nn.softmax`. Other options is `hardmax`, which is
            hardmax() within this module. Any other value will result into
            validation error. Default to use `softmax`.
          kernel_initializer: (optional), the name of the initializer for the
            attention kernel.
          dtype: The data type for the query and memory layers of the attention
            mechanism.
          name: Name to use when creating ops.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        self.probability_fn_name = probability_fn
        probability_fn = self._process_probability_fn(self.probability_fn_name)
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        if dtype is None:
            dtype = tf.float32
        query_layer = kwargs.pop("query_layer", None)
        if not query_layer:
            query_layer = tf.keras.layers.Dense(
                units, name="query_layer", use_bias=False, dtype=dtype)
        memory_layer = kwargs.pop("memory_layer", None)
        if not memory_layer:
            memory_layer = tf.keras.layers.Dense(
                units, name="memory_layer", use_bias=False, dtype=dtype)
        self.units = units
        self.normalize = normalize
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.attention_v = None
        self.attention_g = None
        self.attention_b = None
        super(BahdanauAttention, self).__init__(
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            query_layer=query_layer,
            memory_layer=memory_layer,
            probability_fn=wrapped_probability_fn,
            name=name,
            dtype=dtype,
            **kwargs)

    def build(self, input_shape):
        super(BahdanauAttention, self).build(input_shape)
        if self.attention_v is None:
            self.attention_v = self.add_weight(
                "attention_v", [self.units],
                dtype=self.dtype,
                initializer=self.kernel_initializer)
        if (self.normalize and self.attention_g is None
                and self.attention_b is None):
            self.attention_g = self.add_weight(
                "attention_g",
                initializer=tf.compat.v1.constant_initializer(
                    math.sqrt((1. / self.units))),
                shape=())
            self.attention_b = self.add_weight(
                "attention_b",
                shape=[self.units],
                initializer=tf.zeros_initializer())
        self.built = True

    def _calculate_attention(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
          next_state: same as alignments.
        """
        processed_query = self.query_layer(
            query) if self.query_layer else query
        score = _bahdanau_score(
            processed_query,
            self.keys,
            self.attention_v,
            attention_g=self.attention_g,
            attention_b=self.attention_b)
        alignments = self.probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

    def get_config(self):
        # yapf: disable
        config = {
            "units": self.units,
            "normalize": self.normalize,
            "probability_fn": self.probability_fn_name,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer)
        }
        # yapf: enable

        base_config = super(BahdanauAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = _BaseAttentionMechanism.deserialize_inner_layer_from_config(
            config, custom_objects=custom_objects)
        return cls(**config)


def safe_cumprod(x, *args, **kwargs):
    """Computes cumprod of x in logspace using cumsum to avoid underflow.

    The cumprod function and its gradient can result in numerical instabilities
    when its argument has very small and/or zero values.  As long as the
    argument is all positive, we can instead compute the cumulative product as
    exp(cumsum(log(x))).  This function can be called identically to
    tf.cumprod.

    Args:
      x: Tensor to take the cumulative product of.
      *args: Passed on to cumsum; these are identical to those in cumprod.
      **kwargs: Passed on to cumsum; these are identical to those in cumprod.
    Returns:
      Cumulative product of x.
    """
    with tf.name_scope("SafeCumprod"):
        x = tf.convert_to_tensor(x, name="x")
        tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
        return tf.exp(
            tf.cumsum(
                tf.math.log(tf.clip_by_value(x, tiny, 1)), *args, **kwargs))


def monotonic_attention(p_choose_i, previous_attention, mode):
    """Compute monotonic attention distribution from choosing probabilities.

    Monotonic attention implies that the input sequence is processed in an
    explicitly left-to-right manner when generating the output sequence.  In
    addition, once an input sequence element is attended to at a given output
    timestep, elements occurring before it cannot be attended to at subsequent
    output timesteps.  This function generates attention distributions
    according to these assumptions.  For more information, see `Online and
    Linear-Time Attention by Enforcing Monotonic Alignments`.

    Args:
      p_choose_i: Probability of choosing input sequence/memory element i.
        Should be of shape (batch_size, input_sequence_length), and should all
        be in the range [0, 1].
      previous_attention: The attention distribution from the previous output
        timestep.  Should be of shape (batch_size, input_sequence_length).  For
        the first output timestep, preevious_attention[n] should be
        [1, 0, 0, ..., 0] for all n in [0, ... batch_size - 1].
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.
          * 'recursive' uses tf.scan to recursively compute the distribution.
            This is slowest but is exact, general, and does not suffer from
            numerical instabilities.
          * 'parallel' uses parallelized cumulative-sum and cumulative-product
            operations to compute a closed-form solution to the recurrence
            relation defining the attention distribution.  This makes it more
            efficient than 'recursive', but it requires numerical checks which
            make the distribution non-exact.  This can be a problem in
            particular when input_sequence_length is long and/or p_choose_i has
            entries very close to 0 or 1.
          * 'hard' requires that the probabilities in p_choose_i are all either
            0 or 1, and subsequently uses a more efficient and exact solution.

    Returns:
      A tensor of shape (batch_size, input_sequence_length) representing the
      attention distributions for each sequence in the batch.

    Raises:
      ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
    """
    # Force things to be tensors
    p_choose_i = tf.convert_to_tensor(p_choose_i, name="p_choose_i")
    previous_attention = tf.convert_to_tensor(
        previous_attention, name="previous_attention")
    if mode == "recursive":
        # Use .shape[0] when it's not None, or fall back on symbolic shape
        batch_size = tf.compat.dimension_value(
            p_choose_i.shape[0]) or tf.shape(p_choose_i)[0]
        # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_
        # i[-2]]
        shifted_1mp_choose_i = tf.concat(
            [tf.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
        # Compute attention distribution recursively as
        # q[i] = (1 - p_choose_i[i - 1])*q[i - 1] + previous_attention[i]
        # attention[i] = p_choose_i[i]*q[i]
        attention = p_choose_i * tf.transpose(
            tf.scan(
                # Need to use reshape to remind TF of the shape between loop
                # iterations
                lambda x, yz: tf.reshape(yz[0] * x + yz[1], (batch_size,)),
                # Loop variables yz[0] and yz[1]
                [
                    tf.transpose(shifted_1mp_choose_i),
                    tf.transpose(previous_attention)
                ],
                # Initial value of x is just zeros
                tf.zeros((batch_size,))))
    elif mode == "parallel":
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp_choose_i = safe_cumprod(
            1 - p_choose_i, axis=1, exclusive=True)
        # Compute recurrence relation solution
        attention = p_choose_i * cumprod_1mp_choose_i * tf.cumsum(
            previous_attention /
            # Clip cumprod_1mp to avoid divide-by-zero
            tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.),
            axis=1)
    elif mode == "hard":
        # Remove any probabilities before the index chosen last time step
        p_choose_i *= tf.cumsum(previous_attention, axis=1)
        # Now, use exclusive cumprod to remove probabilities after the first
        # chosen index, like so:
        # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
        # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
        # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
        attention = p_choose_i * tf.cumprod(
            1 - p_choose_i, axis=1, exclusive=True)
    else:
        raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
    return attention


def _monotonic_probability_fn(score,
                              previous_alignments,
                              sigmoid_noise,
                              mode,
                              seed=None):
    """Attention probability function for monotonic attention.

    Takes in unnormalized attention scores, adds pre-sigmoid noise to encourage
    the model to make discrete attention decisions, passes them through a
    sigmoid to obtain "choosing" probabilities, and then calls
    monotonic_attention to obtain the attention distribution.  For more
    information, see

    Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784

    Args:
      score: Unnormalized attention scores, shape
        `[batch_size, alignments_size]`
      previous_alignments: Previous attention distribution, shape
        `[batch_size, alignments_size]`
      sigmoid_noise: Standard deviation of pre-sigmoid noise. Setting this
        larger than 0 will encourage the model to produce large attention
        scores, effectively making the choosing probabilities discrete and the
        resulting attention distribution one-hot.  It should be set to 0 at
        test-time, and when hard attention is not desired.
      mode: How to compute the attention distribution.  Must be one of
        'recursive', 'parallel', or 'hard'.  See the docstring for
        `tf.contrib.seq2seq.monotonic_attention` for more information.
      seed: (optional) Random seed for pre-sigmoid noise.

    Returns:
      A `[batch_size, alignments_size]`-shape tensor corresponding to the
      resulting attention distribution.
    """
    # Optionally add pre-sigmoid noise to the scores
    if sigmoid_noise > 0:
        noise = tf.random.normal(tf.shape(score), dtype=score.dtype, seed=seed)
        score += sigmoid_noise * noise
    # Compute "choosing" probabilities from the attention scores
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = tf.cast(score > 0, score.dtype)
    else:
        p_choose_i = tf.sigmoid(score)
    # Convert from choosing probabilities to attention distribution
    return monotonic_attention(p_choose_i, previous_alignments, mode)


class _BaseMonotonicAttentionMechanism(_BaseAttentionMechanism):
    """Base attention mechanism for monotonic attention.

    Simply overrides the initial_alignments function to provide a dirac
    distribution, which is needed in order for the monotonic attention
    distributions to have the correct behavior.
    """

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the monotonic attentions.

        Initializes to dirac distributions, i.e.
        [1, 0, 0, ...memory length..., 0] for all entries in the batch.

        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.

        Returns:
          A `dtype` tensor shaped `[batch_size, alignments_size]`
          (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return tf.one_hot(
            tf.zeros((batch_size,), dtype=tf.int32), max_time, dtype=dtype)


class BahdanauMonotonicAttention(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Bahadanau-style energy function.

    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the
    memory it can't attend to any prior points at subsequence output timesteps.
    It achieves this by using the _monotonic_probability_fn instead of softmax
    to construct its attention distributions.  Since the attention scores are
    passed through a sigmoid, a learnable scalar bias parameter is applied
    after the score function and before the sigmoid.  Otherwise, it is
    equivalent to BahdanauAttention.  This approach is proposed in

    Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784
    """

    def __init__(self,
                 units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 sigmoid_noise=0.,
                 sigmoid_noise_seed=None,
                 score_bias_init=0.,
                 mode="parallel",
                 kernel_initializer="glorot_uniform",
                 dtype=None,
                 name="BahdanauMonotonicAttention",
                 **kwargs):
        """Construct the Attention mechanism.

        Args:
          units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          normalize: Python boolean. Whether to normalize the energy term.
          sigmoid_noise: Standard deviation of pre-sigmoid noise. See the
            docstring for `_monotonic_probability_fn` for more information.
          sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
          score_bias_init: Initial value for score bias scalar. It's
            recommended to initialize this to a negative value when the length
            of the memory is large.
          mode: How to compute the attention distribution. Must be one of
            'recursive', 'parallel', or 'hard'. See the docstring for
            `tf.contrib.seq2seq.monotonic_attention` for more information.
          kernel_initializer: (optional), the name of the initializer for the
            attention kernel.
          dtype: The data type for the query and memory layers of the attention
            mechanism.
          name: Name to use when creating ops.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        # Set up the monotonic probability fn with supplied parameters
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn,
            sigmoid_noise=sigmoid_noise,
            mode=mode,
            seed=sigmoid_noise_seed)
        query_layer = kwargs.pop("query_layer", None)
        if not query_layer:
            query_layer = tf.keras.layers.Dense(
                units, name="query_layer", use_bias=False, dtype=dtype)
        memory_layer = kwargs.pop("memory_layer", None)
        if not memory_layer:
            memory_layer = tf.keras.layers.Dense(
                units, name="memory_layer", use_bias=False, dtype=dtype)
        self.units = units
        self.normalize = normalize
        self.sigmoid_noise = sigmoid_noise
        self.sigmoid_noise_seed = sigmoid_noise_seed
        self.score_bias_init = score_bias_init
        self.mode = mode
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.attention_v = None
        self.attention_score_bias = None
        self.attention_g = None
        self.attention_b = None
        super(BahdanauMonotonicAttention, self).__init__(
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            query_layer=query_layer,
            memory_layer=memory_layer,
            probability_fn=wrapped_probability_fn,
            name=name,
            dtype=dtype,
            **kwargs)

    def build(self, input_shape):
        super(BahdanauMonotonicAttention, self).build(input_shape)
        if self.attention_v is None:
            self.attention_v = self.add_weight(
                "attention_v", [self.units],
                dtype=self.dtype,
                initializer=self.kernel_initializer)
        if self.attention_score_bias is None:
            self.attention_score_bias = self.add_weight(
                "attention_score_bias",
                shape=(),
                dtype=self.dtype,
                initializer=tf.compat.v1.constant_initializer(
                    self.score_bias_init, dtype=self.dtype))
        if (self.normalize and self.attention_g is None
                and self.attention_b is None):
            self.attention_g = self.add_weight(
                "attention_g",
                dtype=self.dtype,
                initializer=tf.compat.v1.constant_initializer(
                    math.sqrt((1. / self.units))),
                shape=())
            self.attention_b = self.add_weight(
                "attention_b", [self.units],
                dtype=self.dtype,
                initializer=tf.zeros_initializer())
        self.built = True

    def _calculate_attention(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        processed_query = self.query_layer(
            query) if self.query_layer else query
        score = _bahdanau_score(
            processed_query,
            self.keys,
            self.attention_v,
            attention_g=self.attention_g,
            attention_b=self.attention_b)
        score += self.attention_score_bias
        alignments = self.probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

    def get_config(self):

        # yapf: disable
        config = {
            "units": self.units,
            "normalize": self.normalize,
            "sigmoid_noise": self.sigmoid_noise,
            "sigmoid_noise_seed": self.sigmoid_noise_seed,
            "score_bias_init": self.score_bias_init,
            "mode": self.mode,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer),
        }
        # yapf: enable

        base_config = super(BahdanauMonotonicAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = _BaseAttentionMechanism.deserialize_inner_layer_from_config(
            config, custom_objects=custom_objects)
        return cls(**config)


class LuongMonotonicAttention(_BaseMonotonicAttentionMechanism):
    """Monotonic attention mechanism with Luong-style energy function.

    This type of attention enforces a monotonic constraint on the attention
    distributions; that is once the model attends to a given point in the
    memory it can't attend to any prior points at subsequence output timesteps.
    It achieves this by using the _monotonic_probability_fn instead of softmax
    to construct its attention distributions.  Otherwise, it is equivalent to
    LuongAttention.  This approach is proposed in

    [Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.](https://arxiv.org/abs/1704.00784)
    """

    def __init__(self,
                 units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 sigmoid_noise=0.,
                 sigmoid_noise_seed=None,
                 score_bias_init=0.,
                 mode="parallel",
                 dtype=None,
                 name="LuongMonotonicAttention",
                 **kwargs):
        """Construct the Attention mechanism.

        Args:
          units: The depth of the query mechanism.
          memory: The memory to query; usually the output of an RNN encoder.
            This tensor should be shaped `[batch_size, max_time, ...]`.
          memory_sequence_length: (optional): Sequence lengths for the batch
            entries in memory.  If provided, the memory tensor rows are masked
            with zeros for values past the respective sequence lengths.
          scale: Python boolean.  Whether to scale the energy term.
          sigmoid_noise: Standard deviation of pre-sigmoid noise.  See the
            docstring for `_monotonic_probability_fn` for more information.
          sigmoid_noise_seed: (optional) Random seed for pre-sigmoid noise.
          score_bias_init: Initial value for score bias scalar.  It's
            recommended to initialize this to a negative value when the length
            of the memory is large.
          mode: How to compute the attention distribution.  Must be one of
            'recursive', 'parallel', or 'hard'.  See the docstring for
            `tf.contrib.seq2seq.monotonic_attention` for more information.
          dtype: The data type for the query and memory layers of the attention
            mechanism.
          name: Name to use when creating ops.
          **kwargs: Dictionary that contains other common arguments for layer
            creation.
        """
        # Set up the monotonic probability fn with supplied parameters
        if dtype is None:
            dtype = tf.float32
        wrapped_probability_fn = functools.partial(
            _monotonic_probability_fn,
            sigmoid_noise=sigmoid_noise,
            mode=mode,
            seed=sigmoid_noise_seed)
        memory_layer = kwargs.pop("memory_layer", None)
        if not memory_layer:
            memory_layer = tf.keras.layers.Dense(
                units, name="memory_layer", use_bias=False, dtype=dtype)
        self.units = units
        self.scale = scale
        self.sigmoid_noise = sigmoid_noise
        self.sigmoid_noise_seed = sigmoid_noise_seed
        self.score_bias_init = score_bias_init
        self.mode = mode
        self.attention_g = None
        self.attention_score_bias = None
        super(LuongMonotonicAttention, self).__init__(
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            query_layer=None,
            memory_layer=memory_layer,
            probability_fn=wrapped_probability_fn,
            name=name,
            dtype=dtype,
            **kwargs)

    def build(self, input_shape):
        super(LuongMonotonicAttention, self).build(input_shape)
        if self.scale and self.attention_g is None:
            self.attention_g = self.add_weight(
                "attention_g", initializer=tf.ones_initializer, shape=())
        if self.attention_score_bias is None:
            self.attention_score_bias = self.add_weight(
                "attention_score_bias",
                shape=(),
                initializer=tf.compat.v1.constant_initializer(
                    self.score_bias_init, self.dtype))
        self.built = True

    def _calculate_attention(self, query, state):
        """Score the query based on the keys and values.

        Args:
          query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
          state: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).

        Returns:
          alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
          next_state: Same as alignments
        """
        score = _luong_score(query, self.keys, self.attention_g)
        score += self.attention_score_bias
        alignments = self.probability_fn(score, state)
        next_state = alignments
        return alignments, next_state

    def get_config(self):
        config = {
            "units": self.units,
            "scale": self.scale,
            "sigmoid_noise": self.sigmoid_noise,
            "sigmoid_noise_seed": self.sigmoid_noise_seed,
            "score_bias_init": self.score_bias_init,
            "mode": self.mode,
        }
        base_config = super(LuongMonotonicAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = _BaseAttentionMechanism.deserialize_inner_layer_from_config(
            config, custom_objects=custom_objects)
        return cls(**config)


class AttentionWrapperState(
        collections.namedtuple(
            "AttentionWrapperState",
            ("cell_state", "attention", "time", "alignments",
             "alignment_history", "attention_state"))):
    """`namedtuple` storing the state of a `AttentionWrapper`.

    Contains:

      - `cell_state`: The state of the wrapped `RNNCell` at the previous time
        step.
      - `attention`: The attention emitted at the previous time step.
      - `time`: int32 scalar containing the current time step.
      - `alignments`: A single or tuple of `Tensor`(s) containing the
         alignments emitted at the previous time step for each attention
         mechanism.
      - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
         containing alignment matrices from all time steps for each attention
         mechanism. Call `stack()` on each to convert to a `Tensor`.
      - `attention_state`: A single or tuple of nested objects
         containing attention mechanism state for each attention mechanism.
         The objects may contain Tensors or TensorArrays.
    """

    def clone(self, **kwargs):
        """Clone this object, overriding components provided by kwargs.

        The new state fields' shape must match original state fields' shape.
        This will be validated, and original fields' shape will be propagated
        to new fields.

        Example:

        ```python
        initial_state = attention_wrapper.get_initial_state(
            batch_size=..., dtype=...)
        initial_state = initial_state.clone(cell_state=encoder_state)
        ```

        Args:
          **kwargs: Any properties of the state object to replace in the
            returned `AttentionWrapperState`.

        Returns:
          A new `AttentionWrapperState` whose properties are the same as
          this one, except any overridden properties as provided in `kwargs`.
        """

        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                if not tf.executing_eagerly():
                    new_shape = tf.shape(new)
                    old_shape = tf.shape(old)
                    with tf.control_dependencies([
                            tf.compat.v1.assert_equal(  # pylint: disable=bad-continuation
                                new_shape,
                                old_shape,
                                data=[new_shape, old_shape])
                    ]):
                        # Add an identity op so that control deps can kick in.
                        return tf.identity(new)
                else:
                    if old.shape.as_list() != new.shape.as_list():
                        raise ValueError(
                            "The shape of the AttentionWrapperState is "
                            "expected to be same as the one to clone. "
                            "self.shape: %s, input.shape: %s" % (old.shape,
                                                                 new.shape))
                    return new
            return new

        return tf.nest.map_structure(
            with_same_shape, self,
            super(AttentionWrapperState, self)._replace(**kwargs))


def _prepare_memory(memory,
                    memory_sequence_length=None,
                    memory_mask=None,
                    check_inner_dims_defined=True):
    """Convert to tensor and possibly mask `memory`.

    Args:
      memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
      memory_mask: `boolean` tensor with shape [batch_size, max_time]. The
        memory should be skipped when the corresponding mask is False.
      check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.

    Returns:
      A (possibly masked), checked, new `memory`.

    Raises:
      ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    memory = tf.nest.map_structure(
        lambda m: tf.convert_to_tensor(m, name="memory"), memory)
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided "
            "at same time.")
    if memory_sequence_length is not None:
        memory_sequence_length = tf.convert_to_tensor(
            memory_sequence_length, name="memory_sequence_length")
    if check_inner_dims_defined:

        def _check_dims(m):
            if not m.get_shape()[2:].is_fully_defined():
                raise ValueError(
                    "Expected memory %s to have fully defined inner dims, "
                    "but saw shape: %s" % (m.name, m.get_shape()))

        tf.nest.map_structure(_check_dims, memory)
    if memory_sequence_length is None and memory_mask is None:
        return memory
    elif memory_sequence_length is not None:
        seq_len_mask = tf.sequence_mask(
            memory_sequence_length,
            maxlen=tf.shape(tf.nest.flatten(memory)[0])[1],
            dtype=tf.nest.flatten(memory)[0].dtype)
    else:
        # For memory_mask is not None
        seq_len_mask = tf.cast(
            memory_mask, dtype=tf.nest.flatten(memory)[0].dtype)

    def _maybe_mask(m, seq_len_mask):
        """Mask the memory based on the memory mask."""
        rank = m.get_shape().ndims
        rank = rank if rank is not None else tf.rank(m)
        extra_ones = tf.ones(rank - 2, dtype=tf.int32)
        seq_len_mask = tf.reshape(
            seq_len_mask, tf.concat((tf.shape(seq_len_mask), extra_ones), 0))
        return m * seq_len_mask

    return tf.nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask),
                                 memory)


def _maybe_mask_score(score,
                      memory_sequence_length=None,
                      memory_mask=None,
                      score_mask_value=None):
    """Mask the attention score based on the masks."""
    if memory_sequence_length is None and memory_mask is None:
        return score
    if memory_sequence_length is not None and memory_mask is not None:
        raise ValueError(
            "memory_sequence_length and memory_mask can't be provided "
            "at same time.")
    if memory_sequence_length is not None:
        message = ("All values in memory_sequence_length must greater than "
                   "zero.")
        with tf.control_dependencies([
                tf.compat.v1.assert_positive(  # pylint: disable=bad-continuation
                    memory_sequence_length,
                    message=message)
        ]):
            memory_mask = tf.sequence_mask(
                memory_sequence_length, maxlen=tf.shape(score)[1])
    score_mask_values = score_mask_value * tf.ones_like(score)
    return tf.where(memory_mask, score, score_mask_values)


def hardmax(logits, name=None):
    """Returns batched one-hot vectors.

    The depth index containing the `1` is that of the maximum logit value.

    Args:
      logits: A batch tensor of logit values.
      name: Name to use when creating ops.
    Returns:
      A batched one-hot tensor.
    """
    with tf.name_scope(name or "Hardmax"):
        logits = tf.convert_to_tensor(logits, name="logits")
        if tf.compat.dimension_value(logits.get_shape()[-1]) is not None:
            depth = tf.compat.dimension_value(logits.get_shape()[-1])
        else:
            depth = tf.shape(logits)[-1]
        return tf.one_hot(tf.argmax(logits, -1), depth, dtype=logits.dtype)


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    """Computes the attention and alignments for a given
    attention_mechanism."""
    if isinstance(attention_mechanism, _BaseAttentionMechanism):
        alignments, next_attention_state = attention_mechanism(
            [cell_output, attention_state])
    else:
        # For other class, assume they are following _BaseAttentionMechanism,
        # which takes query and state as separate parameter.
        alignments, next_attention_state = attention_mechanism(
            cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context_ = tf.matmul(expanded_alignments, attention_mechanism.values)
    context_ = tf.squeeze(context_, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context_], 1))
    else:
        attention = context_

    return attention, alignments, next_attention_state


class AttentionWrapper(tf.keras.layers.AbstractRNNCell):
    """Wraps another `RNNCell` with attention."""

    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 attention_layer=None,
                 attention_fn=None):
        """Construct the `AttentionWrapper`.

        **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped
        in `AttentionWrapper`, then you must ensure that:

        - The encoder output has been tiled to `beam_width` via
          `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
        - The `batch_size` argument passed to the `get_initial_state` method of
          this wrapper is equal to `true_batch_size * beam_width`.
        - The initial state created with `get_initial_state` above contains a
          `cell_state` value containing properly tiled final state from the
          encoder.

        An example:

        ```
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
            encoder_final_state, multiplier=beam_width)
        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            sequence_length, multiplier=beam_width)
        attention_mechanism = MyFavoriteAttentionMechanism(
            num_units=attention_depth,
            memory=tiled_inputs,
            memory_sequence_length=tiled_sequence_length)
        attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
        decoder_initial_state = attention_cell.get_initial_state(
            batch_size=true_batch_size * beam_width, dtype=dtype)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=tiled_encoder_final_state)
        ```

        Args:
          cell: An instance of `RNNCell`.
          attention_mechanism: A list of `AttentionMechanism` instances or a
            single instance.
          attention_layer_size: A list of Python integers or a single Python
            integer, the depth of the attention (output) layer(s). If None
            (default), use the context as attention at each time step.
            Otherwise, feed the context and cell output into the attention
            layer to generate attention at each time step. If
            attention_mechanism is a list, attention_layer_size must be a list
            of the same length. If attention_layer is set, this must be None.
            If attention_fn is set, it must guaranteed that the outputs of
            attention_fn also meet the above requirements.
          alignment_history: Python boolean, whether to store alignment history
            from all time steps in the final output state (currently stored as
            a time major `TensorArray` on which you must call `stack()`).
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, attention:
              tf.concat([inputs, attention], -1)`.
          output_attention: Python bool.  If `True` (default), the output at
            each time step is the attention value.  This is the behavior of
            Luong-style attention mechanisms.  If `False`, the output at each
            time step is the output of `cell`.  This is the behavior of
            Bhadanau-style attention mechanisms.  In both cases, the
            `attention` tensor is propagated to the next time step via the
            state and is used there. This flag only controls whether the
            attention mechanism is propagated up to the next cell in an RNN
            stack or to the top RNN output.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `get_initial_state()`.  Note that if this value is
            provided now, and the user uses a `batch_size` argument of
            `get_initial_state` which does not match the batch size of
            `initial_cell_state`, proper behavior is not guaranteed.
          name: Name to use when creating ops.
          attention_layer: A list of `tf.tf.keras.layers.Layer` instances or a
            single `tf.tf.keras.layers.Layer` instance taking the context
            and cell output as inputs to generate attention at each time step.
            If None (default), use the context as attention at each time step.
            If attention_mechanism is a list, attention_layer must be a list of
            the same length. If attention_layers_size is set, this must be
            None.
          attention_fn: An optional callable function that allows users to
            provide their own customized attention function, which takes input
            (attention_mechanism, cell_output, attention_state,
            attention_layer) and outputs (attention, alignments,
            next_attention_state). If provided, the attention_layer_size should
            be the size of the outputs of attention_fn.

        Raises:
          TypeError: `attention_layer_size` is not None and
            (`attention_mechanism` is a list but `attention_layer_size` is not;
            or vice versa).
          ValueError: if `attention_layer_size` is not None,
            `attention_mechanism` is a list, and its length does not match that
            of `attention_layer_size`; if `attention_layer_size` and
            `attention_layer` are set simultaneously.
        """
        super(AttentionWrapper, self).__init__(name=name)
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = list(attention_mechanism)
            for attention_mechanism in attention_mechanisms:
                if not isinstance(attention_mechanism, AttentionMechanism):
                    raise TypeError(
                        "attention_mechanism must contain only instances of "
                        "AttentionMechanism, saw type: %s" %
                        type(attention_mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(attention_mechanism, AttentionMechanism):
                raise TypeError(
                    "attention_mechanism must be an AttentionMechanism or "
                    "list of multiple AttentionMechanism instances, saw type: "
                    "%s" % type(attention_mechanism).__name__)
            attention_mechanisms = [attention_mechanism]

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: tf.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError("cell_input_fn must be callable, saw type: %s"
                                % type(cell_input_fn).__name__)

        if attention_layer_size is not None and attention_layer is not None:
            raise ValueError(
                "Only one of attention_layer_size and attention_layer "
                "should be set")

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(
                attention_layer_size if isinstance(attention_layer_size, (
                    list, tuple)) else (attention_layer_size,))
            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                    "If provided, attention_layer_size must contain exactly "
                    "one integer per attention_mechanism, saw: %d vs %d" %
                    (len(attention_layer_sizes), len(attention_mechanisms)))
            self._attention_layers = list(
                tf.keras.layers.Dense(
                    attention_layer_size,
                    name="attention_layer",
                    use_bias=False,
                    dtype=attention_mechanisms[i].dtype) for i,
                attention_layer_size in enumerate(attention_layer_sizes))
            self._attention_layer_size = sum(attention_layer_sizes)
        elif attention_layer is not None:
            self._attention_layers = list(
                attention_layer if isinstance(attention_layer, (
                    list, tuple)) else (attention_layer,))
            if len(self._attention_layers) != len(attention_mechanisms):
                raise ValueError(
                    "If provided, attention_layer must contain exactly one "
                    "layer per attention_mechanism, saw: %d vs %d" % (len(
                        self._attention_layers), len(attention_mechanisms)))
            self._attention_layer_size = sum(
                tf.compat.dimension_value(
                    layer.compute_output_shape([
                        None, cell.output_size +
                        tf.compat.dimension_value(mechanism.values.shape[-1])
                    ])[-1]) for layer, mechanism in zip(
                        self._attention_layers, attention_mechanisms))
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                tf.compat.dimension_value(attention_mechanism.values.shape[-1])
                for attention_mechanism in attention_mechanisms)

        if attention_fn is None:
            attention_fn = _compute_attention
        self._attention_fn = attention_fn

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        with tf.name_scope(name or "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = tf.nest.flatten(initial_cell_state)[-1]
                state_batch_size = (tf.compat.dimension_value(
                    final_state_tensor.shape[0])
                                    or tf.shape(final_state_tensor)[0])
                error_message = (
                    "When constructing AttentionWrapper %s: " % self.name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and initial_cell_state.  Are you using "
                    "the BeamSearchDecoder?  You may need to tile your "
                    "initial state via the tf.contrib.seq2seq.tile_batch "
                    "function with argument multiple=beam_width.")
                with tf.control_dependencies(
                        self._batch_size_checks(  # pylint: disable=bad-continuation
                            state_batch_size, error_message)):
                    self._initial_cell_state = tf.nest.map_structure(
                        lambda s: tf.identity(
                            s, name="check_initial_cell_state"),
                        initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [
            tf.compat.v1.assert_equal(
                batch_size,
                attention_mechanism.batch_size,
                message=error_message)
            for attention_mechanism in self._attention_mechanisms
        ]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.

        Which is returned is determined by how the AttentionMechanism(s) were
        passed to the constructor.

        Args:
          seq: A non-empty sequence of items or generator.

        Returns:
          Either the values in the sequence as a tuple if
          AttentionMechanism(s) were passed to the constructor as a sequence
          or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.

        Returns:
          An `AttentionWrapperState` tuple containing shapes used
          by this object.
        """
        return AttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            attention_state=self._item_or_tuple(
                a.state_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                a.alignments_size if self._alignment_history else () for a in
                self._attention_mechanisms))  # sometimes a TensorArray

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.

        **NOTE** Please see the initializer documentation for details of how
        to call `get_initial_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.

        Args:
          inputs: The inputs that will be fed to this cell.
          batch_size: `0D` integer tensor: the batch size.
          dtype: The internal state data type.

        Returns:
          An `AttentionWrapperState` tuple containing zeroed out tensors and,
          possibly, empty `TensorArray` objects.

        Raises:
          ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        with tf.name_scope(type(self).__name__ + "ZeroState"):  # pylint: disable=bad-continuation
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.get_initial_state(
                    batch_size=batch_size, dtype=dtype)
            error_message = (
                "When calling get_initial_state of AttentionWrapper %s: " %
                self.name + "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size. Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output "
                "has been tiled to beam_width via "
                "tf.contrib.seq2seq.tile_batch, and the batch_size= argument "
                "passed to get_initial_state is batch_size * beam_width.")
            with tf.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):  # pylint: disable=bad-continuation
                cell_state = tf.nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            initial_alignments = [
                attention_mechanism.initial_alignments(batch_size, dtype)
                for attention_mechanism in self._attention_mechanisms
            ]
            return AttentionWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=_zero_state_tensors(self._attention_layer_size,
                                              batch_size, dtype),
                alignments=self._item_or_tuple(initial_alignments),
                attention_state=self._item_or_tuple(
                    attention_mechanism.initial_state(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                    tf.TensorArray(
                        dtype,
                        size=0,
                        dynamic_size=True,
                        element_shape=alignment.shape) if self.
                    _alignment_history else ()
                    for alignment in initial_alignments))

    def call(self, inputs, state, **kwargs):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous
          state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell
          output and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time
            step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
          **kwargs: Dict, other keyword arguments for the cell call method.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError(
                "Expected state to be instance of AttentionWrapperState. "
                "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(
            cell_inputs, cell_state, **kwargs)

        cell_batch_size = (tf.compat.dimension_value(cell_output.shape[0])
                           or tf.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):  # pylint: disable=bad-continuation
            cell_output = tf.identity(cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = self._attention_fn(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state
