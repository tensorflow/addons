# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tensorflow op performing correlation cost operation."""

import tensorflow as tf
from typeguard import typechecked
from tensorflow_addons.utils.resource_loader import LazySO
from tensorflow.python.keras.utils import conv_utils

_deformable_conv2d_ops_so = LazySO("custom_ops/layers/_deformable_conv2d_ops.so")
# _deformable_conv2d_ops_so = LazySO(
#    "/home/admin-seu/TempData/sss/custom_ops/deformable_conv2d_ops_new/deformable_conv2D.so"
# )
# _deformable_conv2d_ops_so = LazySO("/home/admin-seu/TempData/sss/SoftWare/addons/bazel-bin/tensorflow_addons/custom_ops/layers/_deformable_conv2d_ops.so")
# _deformable_conv2d_ops_so = tf.load_op_library("custom_ops/layers/_deformable_conv2d_ops.so")


def _deformable_conv2d(
    input,
    filter,
    offset,
    mask,
    strides=[1, 1, 1, 1],
    num_groups=1,
    deformable_groups=1,
    im2col_step=1,
    no_bias=True,
    padding="VALID",
    data_format="NHWC",
    dilations=[1, 1, 1, 1],
):
    if data_format == "NHWC":
        input = tf.transpose(input, [0, 3, 1, 2])
        filter = tf.transpose(filter, [3, 2, 0, 1])
        offset = tf.transpose(offset, [0, 3, 1, 2])
        mask = tf.transpose(mask, [0, 3, 1, 2])
    ret = _deformable_conv2d_ops_so.ops.addons_deformable_conv2d(
        input=input,
        filter=filter,
        offset=offset,
        mask=mask,
        strides=strides,
        num_groups=num_groups,
        deformable_groups=deformable_groups,
        im2col_step=im2col_step,
        no_bias=no_bias,
        padding=padding,
        data_format="NCHW",
        dilations=dilations,
    )
    if data_format == "NHWC":
        return tf.transpose(ret, [0, 2, 3, 1])
    return ret


@tf.RegisterGradient("AddonsDeformableConv2D")
def _deformable_conv2d_back_prop(op, grad):
    """The gradients for `deform_conv`.
        Args:
        op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `roi_pool` op.
        Returns:
        Gradients with respect to the input of `deform_conv`.
    """
    data = op.inputs[0]
    filter = op.inputs[1]
    offset = op.inputs[2]
    mask = op.inputs[3]
    """
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: {'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    """
    strides = op.get_attr("strides")
    dilations = op.get_attr("dilations")
    data_format = op.get_attr("data_format")
    im2col_step = op.get_attr("im2col_step")
    no_bias = op.get_attr("no_bias")
    pads = op.get_attr("padding")
    num_groups = op.get_attr("num_groups")
    deformable_groups = op.get_attr("deformable_groups")
    """
    REGISTER_OP("Addons>DeformableConv2DBackProp")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Input("mask: T")
        .Input("out_grad: T")
        .Output("x_grad: T")
        .Output("filter_grad: T")
        .Output("offset_grad: T")
        .Output("mask_grad: T")
        .Attr("T: {float, double}")
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: { 'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    """
    # compute gradient
    data_grad = _deformable_conv2d_ops_so.ops.addons_deformable_conv2d_back_prop(
        data,
        filter,
        offset,
        mask,
        grad,
        strides=strides,
        num_groups=num_groups,
        deformable_groups=deformable_groups,
        im2col_step=im2col_step,
        no_bias=no_bias,
        padding=pads,
        data_format=data_format,
        dilations=dilations,
    )
    return data_grad  # List of 4 Tensor, since we have 4 input


# @tf.keras.utils.register_keras_serializable(package="Addons")
class DeformableConv2D(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        filters: int,
        kernel_size: tuple = (3, 3),
        num_groups: int = 1,
        deformable_groups: int = 1,
        strides: tuple = (1, 1),
        im2col: int = 1,
        use_bias: bool = False,
        padding: str = "valid",
        data_format: str = "channels_last",
        dilations: tuple = (1, 1),
    ):
        super(DeformableConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.deformable_groups = deformable_groups
        self.strides = strides
        self.im2col = im2col
        self.use_bias = use_bias
        self.padding = padding
        self.data_format = data_format
        self.dilations = dilations
        self.conv_offset = tf.keras.layers.Conv2D(
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding=self.padding,
            use_bias=True,
            data_format=data_format,
        )

    def build(self, input_shape):
        if self.data_format == "channels_last":
            channel = int(input_shape[-1])
        else:
            channel = int(input_shape[1])
        if self.data_format == "channels_last":
            self.filter = tf.Variable(
                initial_value=tf.random.normal(
                    shape=[
                        self.kernel_size[0],
                        self.kernel_size[1],
                        channel,
                        self.filters,
                    ]
                )
            )
        else:
            self.filter = tf.Variable(
                initial_value=tf.random.normal(
                    shape=[
                        self.filters,
                        channel,
                        self.kernel_size[0],
                        self.kernel_size[1],
                    ]
                )
            )
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters] + new_space)

    def call(self, inputs, **kwargs):
        """
        Build static Graph
        :param inputs: [B, Height, Width, Channel]
        :param kwargs:
        :return:
        """
        weight_info = self.conv_offset(inputs)
        tf_data_format = "NCHW"
        tf_padding = "VALID"
        if self.padding == "same":
            tf_padding = "SAME"
        if self.data_format == "channels_last":
            tf_data_format = "NHWC"
            o1, o2, mask = tf.split(weight_info, 3, axis=-1)
            offset = tf.concat((o1, o2), axis=-1)
            mask = tf.sigmoid(mask)
        else:
            o1, o2, mask = tf.split(weight_info, 3, axis=1)
            offset = tf.concat((o1, o2), axis=1)
            mask = tf.sigmoid(mask)
        result = _deformable_conv2d(
            input=inputs,
            filter=self.filter,
            offset=offset,
            mask=mask,
            strides=[1, self.strides[0], self.strides[1], 1],
            num_groups=self.num_groups,
            deformable_groups=self.deformable_groups,
            im2col_step=self.im2col,
            no_bias=(not self.use_bias),
            padding=tf_padding,
            data_format=tf_data_format,
            dilations=[1, self.dilations[0], self.dilations[1], 1],
        )
        return result

    def get_config(self):
        config = {
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "num_groups": self.num_groups,
            "deformable_groups": self.deformable_groups,
            "strides": self.strides,
            "im2col": self.im2col,
            "use_bias": self.use_bias,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilations": self.dilations,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.RegisterGradient("AddonsDeformablePsroiPool")
def _deformable_psroi_pool_back_prop(op, *grad):
    data = op.inputs[0]
    bbox = op.inputs[1]
    trans = op.inputs[2]
    top_count = op.outputs[1]
    pooled_size = op.get_attr("pooled_size")
    no_trans = op.get_attr("no_trans")
    spatial_scale = op.get_attr("spatial_scale")
    output_dim = op.get_attr("output_dim")
    group_size = op.get_attr("group_size")
    part_size = op.get_attr("part_size")
    sample_per_part = op.get_attr("sample_per_part")
    trans_std = op.get_attr("trans_std")
    data_grad = _deformable_conv2d_ops_so.ops.addons_deformable_psroi_pool_back_prop(
        data,
        bbox,
        trans,
        top_count,
        grad[0],
        pooled_size=pooled_size,
        no_trans=no_trans,
        spatial_scale=spatial_scale,
        output_dim=output_dim,
        group_size=group_size,
        part_size=part_size,
        sample_per_part=sample_per_part,
        trans_std=trans_std,
    )
    return [data_grad[0], tf.zeros_like(bbox), data_grad[1]]


# @tf.keras.utils.register_keras_serializable(package="Addons")
class DeformablePSROIAlign(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim: int = 256,
        spatial_scale: float = 1 / 16,
        group_size: int = 1,
        pooled_size: int = 7,
        sample_per_part: int = 4,
        part_size: int = 7,
        trans_std: int = 1,
        data_format: str = "channels_last",
    ):
        super(DeformablePSROIAlign, self).__init__()
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.output_dim = output_dim
        self.pooled_size = pooled_size
        self.sample_per_part = sample_per_part
        self.part_size = part_size
        self.trans_std = trans_std
        self.data_format = data_format
        self.flat = tf.keras.layers.Flatten(data_format="channels_first")
        self.fully_connect = tf.keras.layers.Dense(
            self.pooled_size * self.pooled_size * 2
        )

    def compute_output_shape(self, input_shape):
        data_shape = input_shape[0]
        batch_size = data_shape[0]
        if self.data_format == "channels_last":
            return tf.TensorShape(
                [batch_size, self.pooled_size, self.pooled_size, self.output_dim]
            )
        else:
            return tf.TensorShape(
                [batch_size, self.output_dim, self.pooled_size, self.pooled_size]
            )

    def call(self, inputs, **kwargs):
        featuremap = inputs[0]
        rois = inputs[1]
        if self.data_format == "channels_last":
            featuremap = tf.transpose(featuremap, perm=[0, 3, 1, 2])
        (
            offset_t,
            top_count,
        ) = _deformable_conv2d_ops_so.ops.addons_deformable_psroi_pool(
            featuremap,
            rois,
            tf.convert_to_tensor(0.0),
            pooled_size=self.pooled_size,
            no_trans=True,
            spatial_scale=self.spatial_scale,
            output_dim=self.output_dim,
            group_size=self.group_size,
            part_size=self.part_size,
            sample_per_part=self.sample_per_part,
            trans_std=1.0,
        )
        offset_flat = self.flat(offset_t)
        offset = self.fully_connect(offset_flat)
        offset_reshape = tf.reshape(offset, shape=[-1, 2, 7, 7], name="offset_reshape")
        ret, ret_count = _deformable_conv2d_ops_so.ops.addons_deformable_psroi_pool(
            featuremap,
            rois,
            offset_reshape,
            pooled_size=self.pooled_size,
            no_trans=False,
            spatial_scale=self.spatial_scale,
            output_dim=self.output_dim,
            group_size=self.group_size,
            part_size=self.part_size,
            sample_per_part=self.sample_per_part,
            trans_std=self.trans_std,
        )
        if self.data_format == "channels_last":
            ret = tf.transpose(ret, [0, 2, 3, 1])
        return ret

    def get_config(self):
        config = {
            "spatial_scale": self.spatial_scale,
            "group_size": self.group_size,
            "output_dim": self.output_dim,
            "pooled_size": self.pooled_size,
            "sample_per_part": self.sample_per_part,
            "part_size": self.part_size,
            "trans_std": self.trans_std,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return {**config, **base_config}
