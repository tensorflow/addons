# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Keras octave convolution layers"""

import abc
import warnings

import tensorflow as tf

from tensorflow.keras import activations
from tensorflow.keras import backend
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer

from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import UpSampling1D

from tensorflow.keras.layers import Conv1D

import tensorflow_addons.utils.keras_utils as conv_utils


class OctaveConv(Layer):
    """
    Abstract N-D octave convolution layer (private, used as implementation base)

    The octave convolutions factorize convolutional feature maps into two groups
    at different spatial frequencies and process them with different
    convolutions at their corresponding frequency, one octave apart.
    This layer creates 4 convolution layers, 2 for high frequency feature maps
    and 2 for low frequency feature maps. For each frequency, the outputs of the
    convolution layers are concatenated in order to get 2 final outputs, with the
    ratio of low frequency feature maps being `low_freq_ratio` and
    (1 - `low_freq_ratio`) for the high frequency feature maps.
    If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments
      rank: An integer, the rank of the convolution, e.g. "2" for 2D
        convolution.
      filters: Integer, the dimensionality of the output space (i.e. the
        number of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      data_format: A string.
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the
        default initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked
        as trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.

    References
      - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
         Networks with Octave Convolution]
        (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        octave=2,
        low_freq_ratio=0.25,
        strides=1,
        padding="same",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        **kwargs
    ):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, "kernel_size")
        self.octave = octave
        self.low_freq_ratio = low_freq_ratio
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, "dilation_rate"
        )
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.filters_low = int(filters * self.low_freq_ratio)
        self.filters_high = filters - self.filters_low

        self.pooling = None
        self.up_sampling = None

        if self.padding != "same":
            warnings.warn(
                "Padding set to {} for the octave convolution layer "
                "with name {}. "
                "For an optimal use of octave convolutions, set "
                "padding to same.".format(self.padding, self.name)
            )

        self.kernel, self.bias = [], []
        self.conv_high_to_high, self.conv_low_to_high = None, None
        self.conv_low_to_low, self.conv_high_to_low = None, None
        self.generate_convolutions()

    @abc.abstractmethod
    def _init_conv(self, filters, name):
        pass

    def generate_convolutions(self):
        if self.filters_high > 0:
            self.conv_high_to_high = self._init_conv(
                self.filters_high, name="{}-Conv{}D-HH".format(self.name, self.rank)
            )
            self.conv_low_to_high = self._init_conv(
                self.filters_high, name="{}-Conv{}D-LH".format(self.name, self.rank)
            )
        if self.filters_low > 0:
            self.conv_low_to_low = self._init_conv(
                self.filters_low, name="{}-Conv{}D-LL".format(self.name, self.rank)
            )
            self.conv_high_to_low = self._init_conv(
                self.filters_low, name="{}-Conv{}D-HL".format(self.name, self.rank)
            )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None

        if input_shape_low is None:
            self.conv_low_to_high, self.conv_low_to_low = None, None

        if self.conv_high_to_high is not None:
            with backend.name_scope(self.conv_high_to_high.name):
                self.conv_high_to_high.build(input_shape_high)
                self.kernel.append(self.conv_high_to_high.kernel)
                self.bias.append(self.conv_high_to_high.bias)
        if self.conv_low_to_high is not None:
            with backend.name_scope(self.conv_low_to_high.name):
                self.conv_low_to_high.build(input_shape_low)
                self.kernel.append(self.conv_low_to_high.kernel)
                self.bias.append(self.conv_low_to_high.bias)
        if self.conv_high_to_low is not None:
            with backend.name_scope(self.conv_high_to_low.name):
                self.conv_high_to_low.build(input_shape_high)
                self.kernel.append(self.conv_high_to_low.kernel)
                self.bias.append(self.conv_high_to_low.bias)
        if self.conv_low_to_low is not None:
            with backend.name_scope(self.conv_low_to_low.name):
                self.conv_low_to_low.build(input_shape_low)
                self.kernel.append(self.conv_low_to_low.kernel)
                self.bias.append(self.conv_low_to_low.bias)

        self.built = True

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs_high, inputs_low = inputs
        else:
            inputs_high, inputs_low = inputs, None

        outputs_high_to_high, outputs_low_to_high = 0.0, 0.0
        if self.conv_high_to_high is not None:
            outputs_high_to_high = self.conv_high_to_high(inputs_high)
        if self.conv_low_to_high is not None:
            outputs_low_to_high = self.up_sampling(self.conv_low_to_high(inputs_low))
        outputs_high = outputs_high_to_high + outputs_low_to_high

        outputs_low_to_low, outputs_high_to_low = 0.0, 0.0
        if self.conv_low_to_low is not None:
            outputs_low_to_low = self.conv_low_to_low(inputs_low)
        if self.conv_high_to_low is not None:
            outputs_high_to_low = self.conv_high_to_low(self.pooling(inputs_high))
        outputs_low = outputs_low_to_low + outputs_high_to_low

        if self.filters_low == 0:
            return outputs_high
        if self.filters_high == 0:
            return outputs_low
        return [outputs_high, outputs_low]

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high = input_shape

        output_shape_high = None
        if self.filters_high > 0:
            # outputs_high is the sum of outputs_high_to_high with
            # outputs_low_to_high so we only need to compute the output shape
            # of either one of them (output_high_to_high in this case)
            output_shape_high = self.conv_high_to_high.compute_output_shape(
                input_shape_high
            )
        output_shape_low = None
        if self.filters_low > 0:
            # outputs_low is the sum of outputs_high_to_low with
            # outputs_low_to_low so we only need to compute the output shape
            # of either one of them (output_high_to_low in this case)
            output_shape_low = self.conv_high_to_low.compute_output_shape(
                self.pooling.compute_output_shape(input_shape_high),
            )

        if self.filters_low == 0:
            return output_shape_high
        if self.filters_high == 0:
            return output_shape_low
        return [output_shape_high, output_shape_low]

    def get_config(self):
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "octave": self.octave,
            "low_freq_ratio": self.low_freq_ratio,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="Addons")
class OctaveConv1D(OctaveConv):
    """1D octave convolution layer (e.g. temporal convolution).

    This layer creates 4 1D-convolution layers that produce 2 tensors of
    outputs (see the documentation of OctaveConv for more information).
    If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model, provide the keyword
    argument `input_shape` (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes with a single
    channel, in `data_format="channels_last"`.

    Examples:

    >>> # The inputs are 128-length vectors with 10 timesteps, and the batch size
    >>> # is None.
    >>> x = Input(shape=(10,128,))
    >>> y = tf.keras.layers.octave_convolutional.OctaveConv1D(32, 3,
    ... padding='same', activation='relu',low_freq_ratio=0.25)(x)
    >>> print(len(y))
    2
    >>> print(y[0].shape, y[1].shape])
    (None, 10, 24) (None, 5, 8)

    Arguments
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      octave: the reduction factor of the spatial dimensions. It must be a
        power of 2.
      low_freq_ratio: The ratio of filters for lower spatial resolution.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: Only `"same"` is considered for octave convolutions
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.

     Input shape:
      First case,
        single input (e.g. first octave convolution layer of the
        architecture):
          3D tensor with shape:
          `(samples, channels, input_dim)` if data_format='channels_first'
          or 3D tensor with shape:
          `(samples, input_dim, channels) if data_format='channels_last'.
      Second case,
        list of two 3D tensors with shape:
          [`(samples, (1-ratio_out) * filters, input_dim_H)`,
          `(samples, ratio_out * filters, input_dim_L)`] if
          data_format='channels_first'
         or list of two 3D tensors with shape:
          [`(samples, input_dim_H, (1-ratio_out) * filters)`,
          `(samples, input_dim_H, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Output shape:
      First case,
        single output (e.g. last octave convolution layer of the
        architecture):
          3D tensor with shape:
          `(samples, channels, output_dim)` if data_format='channels_first'
          or 3D tensor with shape:
          `(samples, output_dim, channels)` if data_format='channels_last'.
      Second case,
        list of two 3D tensors with shape:
          [`(samples, (1-ratio_out) * filters, output_dim_H)`,
          `(samples, ratio_out * filters, output_dim_L)`] if
          data_format='channels_first'
        or list of two 3D tensors with shape:
          [`(samples, output_dim_H, (1-ratio_out) * filters)`,
          `(samples, output_dim_L, ratio_out * filters)`] if
          data_format='channels_last'
          suffixes _H for high frequency feature maps and _L for low frequency
          feature maps

    Raises:
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.

    References
        - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural
           Networks with Octave Convolution]
          (https://arxiv.org/pdf/1904.05049.pdf)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        octave=2,
        rank=1,
        low_freq_ratio=0.25,
        strides=1,
        padding="same",
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            octave=octave,
            low_freq_ratio=low_freq_ratio,
            strides=strides,
            padding=padding,
            data_format=conv_utils.normalize_data_format(data_format),
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.pooling = AveragePooling1D(
            pool_size=self.octave,
            padding="valid",
            data_format=data_format,
            name="{}-AveragePooling1D".format(self.name),
        )
        self.up_sampling = UpSampling1D(
            size=self.octave, name="{}-UpSampling1D".format(self.name),
        )

    def _init_conv(self, filters, name):
        return Conv1D(
            filters=filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            trainable=self.trainable,
            name=name,
        )

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape_high, input_shape_low = input_shape
        else:
            input_shape_high, input_shape_low = input_shape, None
        if len(input_shape_high) != 3:
            raise ValueError(
                "High frequency input should have rank 3; Received "
                "input shape {}".format(str(input_shape_high))
            )
        if self.data_format == "channels_first":
            channel_axis, data_axis = 1, 2
        else:
            data_axis, channel_axis = 1, 2
        if input_shape_high[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the higher spatial inputs "
                "should be defined. Found `None`."
            )
        if input_shape_low is not None and input_shape_low[channel_axis] is None:
            raise ValueError(
                "The channel dimension of the lower spatial inputs "
                "should be defined. Found `None`."
            )
        if (
            input_shape_high[data_axis] is not None
            and input_shape_high[data_axis] % self.octave != 0
        ):
            raise ValueError(
                "The dimension with the data of the higher spatial inputs "
                "should be divisible by the octave. "
                "Found {} and {}.".format(input_shape_high, self.octave)
            )

        super().build(input_shape)
