# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import typing

import tensorflow as tf
from typeguard import typechecked
from tensorflow_addons.utils import types
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow.python.keras.utils import conv_utils

_deformable_conv2d_ops_so = LazySO("custom_ops/layers/_deformable_conv2d_ops.so")


@typechecked
def _deformable_conv2d(
    input_tensor: tf.Tensor,
    filter_tensor: tf.Tensor,
    bias_tensor: tf.Tensor,
    offset_tensor: tf.Tensor,
    mask_tensor: tf.Tensor,
    strides: typing.Union[tuple, list],
    dilations: typing.Union[tuple, list],
    weight_groups: int,
    offset_groups: int,
    padding: str,
):
    with tf.name_scope("deformable_conv2d"):
        return _deformable_conv2d_ops_so.ops.addons_deformable_conv2d(
            input=input_tensor,
            filter=filter_tensor,
            bias=bias_tensor,
            offset=offset_tensor,
            mask=mask_tensor,
            strides=strides,
            weight_groups=weight_groups,
            offset_groups=offset_groups,
            padding=padding,
            data_format="NCHW",
            dilations=dilations,
        )


@tf.RegisterGradient("Addons>DeformableConv2D")
def _deformable_conv2d_grad(op, grad):
    input = op.inputs[0]
    filter = op.inputs[1]
    bias = op.inputs[2]
    offset = op.inputs[3]
    mask = op.inputs[4]
    strides = op.get_attr("strides")
    weight_groups = op.get_attr("weight_groups")
    offset_groups = op.get_attr("offset_groups")
    padding = op.get_attr("padding")
    dilations = op.get_attr("dilations")
    data_format = op.get_attr("data_format")

    data_grad = _deformable_conv2d_ops_so.ops.addons_deformable_conv2d_grad(
        input,
        filter,
        bias,
        offset,
        mask,
        grad,
        strides=strides,
        weight_groups=weight_groups,
        offset_groups=offset_groups,
        padding=padding,
        dilations=dilations,
        data_format=data_format,
    )
    return data_grad


@tf.keras.utils.register_keras_serializable(package="Addons")
class DeformableConv2D(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        filters: int,
        kernel_size: typing.Union[int, tuple, list] = (3, 3),
        strides: typing.Union[int, tuple, list] = (1, 1),
        padding: str = "valid",
        data_format: str = "channels_first",
        dilation_rate: typing.Union[int, tuple, list] = (1, 1),
        weight_groups: int = 1,
        offset_groups: int = 1,
        use_mask: bool = False,
        use_bias: bool = False,
        kernel_initializer: types.Initializer = None,
        bias_initializer: types.Initializer = None,
        kernel_regularizer: types.Regularizer = None,
        bias_regularizer: types.Regularizer = None,
        kernel_constraint: types.Constraint = None,
        bias_constraint: types.Constraint = None,
        **kwargs
    ):
        """Modulated Deformable Convolution Layer.

        This layer implements from [Deformable ConvNets v2: More Deformable, Better Results]
        (https://arxiv.org/abs/1811.11168)(Zhu et al.).

        Arguments:
          filters: Integer, the dimensionality of the output space (i.e. the number of
            output filters in the convolution).
          kernel_size: An integer or tuple/list of 2 integers, specifying the height
            and width of the 2D convolution window. Can be a single integer to specify
            the same value for all spatial dimensions.
          strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width. Can be a single integer to
            specify the same value for all spatial dimensions. Specifying any stride
            value != 1 is incompatible with specifying any `dilation_rate` value != 1.
          padding: one of `"valid"` or `"same"` (case-insensitive).
          data_format: Specifies the data format.
            Possible values is:
                "channels_first" float [batch, channels, height, width]
                Defaults to `"channels_first"`.
          dilation_rate: an integer or tuple/list of 2 integers, specifying the
            dilation rate to use for dilated convolution. Can be a single integer to
            specify the same value for all spatial dimensions.
          weight_groups: A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved separately
            with `filters / weight_groups` filters. The output is the concatenation of all
            the `weight_groups` results along the channel axis. Input channels and `filters`
            must both be divisible by `groups`.
          offset_groups: An integer specifying the number of groups in which the input is
            split along the channel axis. Each group is convolved separately with
            its group offset.
          use_mask: Boolean, whether the layer uses a modulation input.
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix (see
            `keras.initializers`).
          bias_initializer: Initializer for the bias vector (see
            `keras.initializers`).
          kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix (see `keras.regularizers`).
          bias_regularizer: Regularizer function applied to the bias vector (see
            `keras.regularizers`).
          activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation") (see `keras.regularizers`).
          kernel_constraint: Constraint function applied to the kernel matrix (see
            `keras.constraints`).
          bias_constraint: Constraint function applied to the bias vector (see
            `keras.constraints`).
        """
        super(DeformableConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = conv_utils.normalize_tuple(strides, 2, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, 2, "dilation_rate"
        )
        self.weight_groups = weight_groups
        self.offset_groups = offset_groups
        self.use_mask = use_mask
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if self.padding == "causal":
            raise ValueError("Causal padding is not supported.")

        if self.data_format != "channels_first":
            raise ValueError("`channels_last` data format is not supported.")

        if self.filters % self.weight_groups != 0:
            raise ValueError("filters must be divisible by weight_group.")

        self.filter_weights = None
        self.filter_bias = None

    def _validate_shapes(self, shapes):
        if type(shapes) is not list:
            raise ValueError("DeformableConv2D input must be list of Tensor.")
        elif self.use_mask and len(shapes) != 3:
            raise ValueError("DeformableConv2D input must be 3-length list of Tensor.")
        elif not self.use_mask and len(shapes) != 2:
            raise ValueError("DeformableConv2D input must be 2-length list of Tensor.")

    def build(self, shapes):
        self._validate_shapes(shapes)

        input_shape = shapes[0]
        offset_shape = shapes[1]
        mask_shape = shapes[2] if self.use_mask else None

        exp_off_c = self.offset_groups * 2 * self.kernel_size[0] * self.kernel_size[1]

        off_b, off_c, off_h, off_w = offset_shape
        in_b, in_c, in_h, in_w = input_shape

        out_h = conv_utils.conv_output_length(
            in_h,
            self.kernel_size[0],
            padding=self.padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0],
        )
        out_w = conv_utils.conv_output_length(
            in_w,
            self.kernel_size[1],
            padding=self.padding,
            stride=self.strides[1],
            dilation=self.dilation_rate[1],
        )

        if off_b != in_b or off_c != exp_off_c or off_h != out_h or off_w != out_w:
            raise ValueError(
                f"DeformableConv2D Offset shape must be [{in_b}, {exp_off_c}, {out_h}, {out_w}]."
            )

        if mask_shape is not None:
            exp_mask_c = exp_off_c // 2

            mask_b, mask_c, mask_h, mask_w = mask_shape

            if (
                mask_b != in_b
                or mask_c != exp_mask_c
                or mask_h != out_h
                or mask_w != out_w
            ):
                raise ValueError(
                    f"DeformableConv2D Mask shape must be [{in_b}, {exp_mask_c}, {out_h}, {out_w}]."
                )

        # Channel first
        shape = (
            self.filters,
            input_shape[1] // self.weight_groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )

        self.filter_weights = self.add_weight(
            name="filter",
            shape=shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        if self.use_bias:
            self.filter_bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.filter_bias = tf.zeros((0,))

        self.built = True

    def compute_output_shape(self, shapes):
        self._validate_shapes(shapes)

        input_shape = shapes[0]
        in_b, _, in_h, in_w = input_shape

        out_h = conv_utils.conv_output_length(
            in_h,
            self.kernel_size[0],
            padding=self.padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0],
        )
        out_w = conv_utils.conv_output_length(
            in_w,
            self.kernel_size[1],
            padding=self.padding,
            stride=self.strides[1],
            dilation=self.dilation_rate[1],
        )

        return tf.TensorShape([in_b, self.filters, out_h, out_w])

    def call(self, inputs, **kwargs):
        input_tensor = inputs[0]
        offset_tensor = inputs[1]
        mask_tensor = inputs[2] if self.use_mask else tf.zeros((0, 0, 0, 0))

        return _deformable_conv2d(
            input_tensor=tf.convert_to_tensor(input_tensor),
            filter_tensor=tf.convert_to_tensor(self.filter_weights),
            bias_tensor=tf.convert_to_tensor(self.filter_bias),
            offset_tensor=tf.convert_to_tensor(offset_tensor),
            mask_tensor=tf.convert_to_tensor(mask_tensor),
            strides=self.strides,
            weight_groups=self.weight_groups,
            offset_groups=self.offset_groups,
            padding="SAME" if self.padding == "same" else "VALID",
            dilations=self.dilation_rate,
        )

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "weight_groups": self.weight_groups,
            "offset_groups": self.offset_groups,
            "use_mask": self.use_mask,
            "use_bias": self.use_bias,
        }
        base_config = super().get_config()
        return {**base_config, **config}
