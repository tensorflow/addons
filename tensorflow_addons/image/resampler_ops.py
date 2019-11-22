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
"""Python layer for Resampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_resampler_ops = tf.load_op_library(
    get_path_to_datafile("custom_ops/image/_resampler_ops.so"))


@tf.function
def resampler(data, warp, name=None):
    """Resamples input data at user defined coordinates.

    The resampler currently only supports bilinear interpolation of 2D data.
    Args:
      data: Tensor of shape `[batch_size, data_height, data_width,
        data_num_channels]` containing 2D data that will be resampled.
      warp: Tensor of minimum rank 2 containing the coordinates at
      which resampling will be performed. Since only bilinear
      interpolation is currently supported, the last dimension of the
      `warp` tensor must be 2, representing the (x, y) coordinate where
      x is the index for width and y is the index for height.
      name: Optional name of the op.
    Returns:
      Tensor of resampled values from `data`. The output tensor shape
      is determined by the shape of the warp tensor. For example, if `data`
      is of shape `[batch_size, data_height, data_width, data_num_channels]`
      and warp of shape `[batch_size, dim_0, ... , dim_n, 2]` the output will
      be of shape `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
    Raises:
      ImportError: if the wrapper generated during compilation is not
      present when the function is called.
    """
    with tf.name_scope(name or "resampler"):
        data_tensor = tf.convert_to_tensor(data, name="data")
        warp_tensor = tf.convert_to_tensor(warp, name="warp")
        return _resampler_ops.addons_resampler(data_tensor, warp_tensor)


@tf.RegisterGradient("Addons>Resampler")
def _resampler_grad(op, grad_output):
    data, warp = op.inputs
    grad_output_tensor = tf.convert_to_tensor(grad_output, name="grad_output")
    return _resampler_ops.addons_resampler_grad(data, warp, grad_output_tensor)


tf.no_gradient("Addons>ResamplerGrad")
