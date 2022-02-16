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
# =============================================================================

import logging
import copy
from typing import Union, List, Tuple, Iterable
from functools import partial
import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class WeightNormalization(tf.keras.layers.Wrapper):
    """Performs weight normalization.

    This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction.
    This speeds up convergence by improving the
    conditioning of the optimization problem.

    See [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868).

    Wrap `tf.keras.layers.Conv2D`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = WeightNormalization(tf.keras.layers.Conv2D(2, 2), data_init=False)
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])

    Wrap `tf.keras.layers.Dense`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = WeightNormalization(tf.keras.layers.Dense(10), data_init=False)
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])

    Args:
      layer: A `tf.keras.layers.Layer` instance.
      data_init: If `True` use data dependent variable initialization.
    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights.
      NotImplementedError: If `data_init` is True and running graph execution.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, data_init: bool = True, **kwargs):
        super().__init__(layer, **kwargs)
        self.data_init = data_init
        self._track_trackable(layer, name="layer")
        self.is_rnn = isinstance(self.layer, tf.keras.layers.RNN)

        if self.data_init and self.is_rnn:
            logging.warning(
                "WeightNormalization: Using `data_init=True` with RNNs "
                "is advised against by the paper. Use `data_init=False`."
            )

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if not self.layer.built:
            self.layer.build(input_shape)

        kernel_layer = self.layer.cell if self.is_rnn else self.layer

        if not hasattr(kernel_layer, "kernel"):
            raise ValueError(
                "`WeightNormalization` must wrap a layer that"
                " contains a `kernel` for weights"
            )

        if self.is_rnn:
            kernel = kernel_layer.recurrent_kernel
        else:
            kernel = kernel_layer.kernel

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(kernel.shape[-1])
        self.kernel_norm_axes = list(range(kernel.shape.rank - 1))

        self.g = self.add_weight(
            name="g",
            shape=(self.layer_depth,),
            initializer="ones",
            dtype=kernel.dtype,
            trainable=True,
        )
        self.v = kernel

        self._initialized = self.add_weight(
            name="initialized",
            shape=None,
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
        )

        if self.data_init:
            # Used for data initialization in self._data_dep_init.
            with tf.name_scope("data_dep_init"):
                layer_config = tf.keras.layers.serialize(self.layer)
                layer_config["config"]["trainable"] = False
                self._naked_clone_layer = tf.keras.layers.deserialize(layer_config)
                self._naked_clone_layer.build(input_shape)
                self._naked_clone_layer.set_weights(self.layer.get_weights())
                if not self.is_rnn:
                    self._naked_clone_layer.activation = None

        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        def _do_nothing():
            return tf.identity(self.g)

        def _update_weights():
            # Ensure we read `self.g` after _update_weights.
            with tf.control_dependencies(self._initialize_weights(inputs)):
                return tf.identity(self.g)

        g = tf.cond(self._initialized, _do_nothing, _update_weights)

        with tf.name_scope("compute_weights"):
            # Replace kernel by normalized weight variable.
            kernel = tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * g

            if self.is_rnn:
                self.layer.cell.recurrent_kernel = kernel
                update_kernel = tf.identity(self.layer.cell.recurrent_kernel)
            else:
                self.layer.kernel = kernel
                update_kernel = tf.identity(self.layer.kernel)

            # Ensure we calculate result after updating kernel.
            with tf.control_dependencies([update_kernel]):
                outputs = self.layer(inputs)
                return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    def _initialize_weights(self, inputs):
        """Initialize weight g.

        The initial value of g could either from the initial value in v,
        or by the input value if self.data_init is True.
        """
        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(  # pylint: disable=bad-continuation
                    self._initialized, False, message="The layer has been initialized."
                )
            ]
        ):
            if self.data_init:
                assign_tensors = self._data_dep_init(inputs)
            else:
                assign_tensors = self._init_norm()
            assign_tensors.append(self._initialized.assign(True))
            return assign_tensors

    def _init_norm(self):
        """Set the weight g with the norm of the weight vector."""
        with tf.name_scope("init_norm"):
            v_flat = tf.reshape(self.v, [-1, self.layer_depth])
            v_norm = tf.linalg.norm(v_flat, axis=0)
            g_tensor = self.g.assign(tf.reshape(v_norm, (self.layer_depth,)))
            return [g_tensor]

    def _data_dep_init(self, inputs):
        """Data dependent initialization."""
        with tf.name_scope("data_dep_init"):
            # Generate data dependent init values
            x_init = self._naked_clone_layer(inputs)
            data_norm_axes = list(range(x_init.shape.rank - 1))
            m_init, v_init = tf.nn.moments(x_init, data_norm_axes)
            scale_init = 1.0 / tf.math.sqrt(v_init + 1e-10)

            # RNNs have fused kernels that are tiled
            # Repeat scale_init to match the shape of fused kernel
            # Note: This is only to support the operation,
            # the paper advises against RNN+data_dep_init
            if scale_init.shape[0] != self.g.shape[0]:
                rep = int(self.g.shape[0] / scale_init.shape[0])
                scale_init = tf.tile(scale_init, [rep])

            # Assign data dependent init values
            g_tensor = self.g.assign(self.g * scale_init)
            if hasattr(self.layer, "bias") and self.layer.bias is not None:
                bias_tensor = self.layer.bias.assign(-m_init * scale_init)
                return [g_tensor, bias_tensor]
            else:
                return [g_tensor]

    def get_config(self):
        config = {"data_init": self.data_init}
        base_config = super().get_config()
        return {**base_config, **config}

    def remove(self):
        kernel = tf.Variable(
            tf.nn.l2_normalize(self.v, axis=self.kernel_norm_axes) * self.g,
            name="recurrent_kernel" if self.is_rnn else "kernel",
        )

        if self.is_rnn:
            self.layer.cell.recurrent_kernel = kernel
        else:
            self.layer.kernel = kernel

        return self.layer


@tf.keras.utils.register_keras_serializable(package="Addons")
class SpecificConvPad(tf.keras.layers.Wrapper):
    """
    Extend padding behavior of tf.keras.layers.Conv1D, tf.keras.layers.Conv2D
    and tf.keras.layers.Conv3D.

    As everyone knows, convolution layer in keras API only support
    "same", "valid", "causal" and "full" padding. However, these padding
    methods is different in shape-wise but same in numerical-wise, i.e.,
    they are all zero-padding. If we need "reflect" padding, or
    "symmetric" padding, just like 'tf.pad()', it will be very inconvenient.

    This wrapper gives a convolution layer more spcific padding method,
    like "constant" padding with a constant that user can specify, "reflect"
    padding or "symmetric" padding, just like 'tf.pad()', maintain original
    layer's shape-wise behavior ant only change the numerical-wise behavior.
    For example, a user can get a convolution layer with "same" padding in
    shape-wise and "reflect" padding in numerical-wise simultaneously.

    Wrap `tf.keras.layers.Conv1D`:
    >>> import numpy as np
    >>> x = np.random.rand(1, 5, 1)
    >>> conv1d = tf.keras.layers.Conv1D(filters=1, kernel_size=[3,]*1, strides=(1,)*1, padding="same",dilation_rate=1)
    >>> conv1d = SpecificConvPad(conv1d, padding_mode='constant',padding_constant=0)
    >>> y = conv1d(x)
    >>> print(y.shape)
    (1, 5, 1)

    Inidicate the artificial padding behavior:
    >>> x = tf.constant([1.,2.,3.,4.,5.],shape=[1,5,1])
    >>> conv1d_ = tf.keras.layers.Conv1D(filters=1, kernel_size=[2,]*1, strides=(1,)*1, padding="same",dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    >>> conv1d = SpecificConvPad(conv1d_, padding_mode='constant',padding_constant=1)
    >>> y = conv1d(x)
    >>> # x --> padded_x [1.,2.,3.,4.,5.,1.], zero padding x from right side
    >>> # kernel = [1,1]
    >>> # padded_x --> y [3.,7.,5.,9.,6.], conv padded x by kernel, and do not need extral 'padding'
    >>> print(tf.squeeze(y))
    tf.Tensor([3. 5. 7. 9. 6.], shape=(5,), dtype=float32)
    >>> conv1d = SpecificConvPad(conv1d_, padding_mode='reflect')
    >>> y = conv1d(x)
    >>> # x --> padded_x [1.,2.,3.,4.,5.,4.], zero padding x from right side
    >>> # kernel = [1,1]
    >>> # padded_x --> y [3.,7.,5.,9.,9.], conv padded x by kernel, and do not need extral 'padding'
    >>> print(tf.squeeze(y))
    tf.Tensor([3. 5. 7. 9. 9.], shape=(5,), dtype=float32)
    >>> conv1d = SpecificConvPad(conv1d_, padding_mode='symmetric')
    >>> y = conv1d(x)
    >>> # x --> padded_x [1.,2.,3.,4.,5.,5.], zero padding x from right side
    >>> # kernel = [1,1]
    >>> # padded_x --> y [3.,7.,5.,9.,10.], conv padded x by kernel, and do not need extral 'padding'
    >>> print(tf.squeeze(y))
    tf.Tensor([3. 5. 7. 9. 10.], shape=(5,), dtype=float32)

    This wrapper will maintain original layer's shape-wise behavior ant only change the numerical-wise behavior:
    >>> import numpy as np
    >>> x = tf.constant([1.,2.,3.,4.,5.],shape=[1,5,1])
    >>> print(tf.squeeze(x))
    tf.Tensor([1. 2. 3. 4. 5.], shape=(5,), dtype=float32)
    >>> conv1d = tf.keras.layers.Conv1D(filters=1, kernel_size=[3,]*1, strides=(2,)*1, padding="same",dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    >>> conv1d_ = tf.keras.layers.Conv1D(filters=1, kernel_size=[3,]*1, strides=(2,)*1, padding="same",dilation_rate=1,kernel_initializer=tf.initializers.Ones(),use_bias=False)
    >>> conv1d_2 = SpecificConvPad(conv1d_, padding_mode='constant',padding_constant=0)
    When 'constant' padding with constant 0, wrappered layer should have the same behavior than original one. So:
    >>> y = conv1d(x)
    >>> print(tf.squeeze(y))
    tf.Tensor([3. 9. 9.], shape=(3,), dtype=float32) # the original layer's output
    >>> y_2 = conv1d_2(x)
    >>> print(tf.squeeze(y_2))
    tf.Tensor([3. 9. 9.], shape=(3,), dtype=float32) # the wrappered layer's output
    >>> print(np.isclose(tf.reduce_mean(y-y_2),0.0))
    True #

    Args:
      layer: A `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D` or `tf.keras.layers.Conv3D` instance.
      padding_mode: Specific padding mode in `constant`, `reflect` or `symmetric`.
      padding_constant: Padding constant for `constant` padding_mode.
    Raises:
      ValueError: If not initialized with a `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D` or `tf.keras.layers.Conv3D` instance.
      ValueError: If `padding_mode` does not in `constant`, `reflect` or `symmetric`.
      ValueError: If original `layers.padding` is `causal` but `padding_mode` is not `constant`.
    """

    @typechecked
    def __init__(
        self,
        layer: Union[
            tf.keras.layers.Conv1D,
            tf.keras.layers.Conv2D,
            tf.keras.layers.Conv3D,
        ],
        padding_mode: Union[str, None] = None,
        padding_constant: int = 0,
        **kwargs,
    ):

        self.padding_vectors = None
        self.padding_mode = padding_mode
        self.padding_constant = padding_constant

        if "name" not in kwargs.keys():
            kwargs["name"] = "specific_padded_" + layer.name
        super(SpecificConvPad, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])
        # input_shape is needed for
        _kernel_size = self.layer.kernel_size
        _strides = self.layer.strides
        self._padding = (
            _padding
        ) = (
            self.layer.padding.lower()
        )  # save original padding for get_config and from_config
        _tf_data_format = self.layer._tf_data_format
        _dilation_rate = self.layer.dilation_rate
        _input_shape = self._grab_length_by_data_format(
            _tf_data_format, input_shape.as_list()
        )

        if self.padding_mode is None:
            self.fused = True
        else:  # padding_mode is str, confirmed by @typechecked
            if isinstance(_padding, (list, tuple)):
                self.fused = True
            else:
                if _padding in ["valid"]:
                    self.fused = True
                elif _padding in ["same", "full", "causal"]:
                    self.padding_mode = self._normalize_specific_padding_mode(
                        self.padding_mode
                    )
                    if _padding == "causal":
                        if self.padding_mode != self._normalize_specific_padding_mode(
                            "constant"
                        ):
                            raise ValueError(
                                "specific causal padding mode should only be CONSTANT not {}",
                                format(self.padding_mode),
                            )
                    self.fused = False
                    _get_conv_paddings = partial(
                        self._get_conv_paddings, padding=_padding
                    )
                    _paddings = iter(
                        list(
                            map(
                                _get_conv_paddings,
                                _input_shape,
                                _kernel_size,
                                _strides,
                                _dilation_rate,
                            )
                        )
                    )
                    self.padding_vectors = self._norm_paddings_by_data_format(
                        _tf_data_format, _paddings
                    )
                    self.layer.padding = self._normalize_padding("valid")
                    self.layer._is_causal = (
                        self.layer.padding == self._normalize_padding("causal")
                    )  # causal is very special
        layer_input_shape = self._prefix_input_shape(input_shape)
        super().build(layer_input_shape)

    def _prefix_input_shape(self, input_shape):
        if not self.fused:
            input_shape = tf.TensorShape(input_shape).as_list()
            input_shape = copy.deepcopy(input_shape)
            layer_input_shape = list(
                map(
                    self._get_padded_length_from_paddings,
                    input_shape,
                    self.padding_vectors,
                )
            )
        else:
            layer_input_shape = input_shape
        return layer_input_shape

    def call(self, inputs, **kwargs):
        """Call `Layer`"""
        # For correct output when a layer have not been built, pad behavior must put here but not in build() func
        if not self.fused:
            inputs = tf.pad(
                inputs,
                paddings=self.padding_vectors,
                mode=self.padding_mode,
                constant_values=self.padding_constant,
            )
        output = self.layer(inputs, **kwargs)
        return output

    def get_config(self):
        config = {
            "padding_mode": self.padding_mode,
            "padding_constant": self.padding_constant,
        }
        base_config = super().get_config()
        if "layer" in base_config.keys():
            if "config" in base_config["layer"].keys():
                if "padding" in base_config["layer"]["config"].keys():
                    base_config["layer"]["config"]["padding"] = self._padding
        return dict(list(base_config.items()) + list(config.items()))

    @typechecked
    def _normalize_padding(self, value: Union[List, Tuple, str]):
        if isinstance(value, (list, tuple)):
            return value
        padding = value.lower()
        if padding not in {"valid", "same", "causal", "full"}:
            raise ValueError(
                "The `padding` argument must be a list/tuple or one of "
                '"valid", "same", "full" (or "causal", only for `Conv1D). '
                f"Received: {padding}"
            )
        return padding

    @typechecked
    def _normalize_specific_padding_mode(self, value: Union[List, Tuple, str]):
        if isinstance(value, (list, tuple)):
            return value
        padding = value.lower()
        if padding not in {"constant", "reflect", "symmetric"}:
            raise ValueError(
                "The `padding` argument must be a list/tuple or one of "
                '"constant", "reflect" or "symmetric"). '
                f"Received: {padding}"
            )
        return padding.upper()

    @typechecked
    def _get_conv_paddings(
        self,
        input_length: int,
        filter_size: int,
        stride: int,
        dilation_rate: int,
        padding: str,
    ):
        """
        Give out equivalent conv paddings from current padding to VALID padding.
        For example, there is a conv(X,padding='same'), find the equivalent conv paddings and
        make conv(X,padding='same')===conv(pad(X,equivalent_conv_paddings),padding='VALID')
        see the:
        https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
        If padding == "SAME": output_shape = ceil(input_length/stride)
        If padding == "VALID": output_shape = ceil((input_length-(filter_size-1)*dilation_rate)/stride)

        return  paddings(padding vectors) for conv's padding behaviour
        """
        dilated_filter_size = filter_size + (filter_size - 1) * (dilation_rate - 1)
        padding = padding.lower()
        if padding in "valid":
            pad_left = 0
            pad_right = 0
        elif padding == "causal":
            pad_left = dilated_filter_size - 1
            pad_right = 0
        elif padding == "same":
            flag = input_length % stride
            if flag == 0:
                pad_all = max(dilated_filter_size - stride, 0)
            else:
                pad_all = max(dilated_filter_size - flag, 0)
            pad_left = pad_all // 2
            pad_right = pad_all - pad_left
        elif (
            padding == "full"
        ):  # full padding has been deprecated in many conv or deconv layers
            pad_left = dilated_filter_size - 1
            pad_right = dilated_filter_size - 1
        else:
            raise ValueError(
                "Padding should in 'valid', 'causal', 'same' or 'full', not {}.", format
            )
        return [pad_left, pad_right]

    @typechecked
    def _get_padded_length_from_paddings(
        self, length: Union[int, None], paddings: Tuple[int, int]
    ):
        if length is not None:
            pad_left, pad_right = paddings
            length = length + pad_left + pad_right
        return length

    @typechecked
    def _norm_paddings_by_data_format(self, data_format: str, paddings: Iterable):
        out_buf = []
        for data_format_per_dim in data_format:
            if data_format_per_dim.upper() in ["N", "C"]:
                out_buf.append(tuple([0, 0]))
            elif data_format_per_dim.upper() in ["D", "H", "W"]:
                out_buf.append(tuple(next(paddings)))
            else:
                raise ValueError(
                    "data_format should consist with 'N', 'C', 'D', 'H' or 'W' but not '{}'.".format(
                        out_buf
                    )
                )
        return out_buf

    @typechecked
    def _grab_length_by_data_format(self, data_format: str, length: Union[Tuple, List]):
        out_buf = []
        for data_format_per_dim, length_per_dim in zip(data_format, length):
            if data_format_per_dim.upper() in ["N", "C"]:
                pass
            elif data_format_per_dim.upper() in ["D", "H", "W"]:
                out_buf.append(int(length_per_dim))
            else:
                raise ValueError(
                    "data_format should consist with 'N', 'C', 'D', 'H' or 'W' but not '{}'.".format(
                        out_buf
                    )
                )
        return out_buf
