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

import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.deformable_conv2d import (
    DeformableConv2D,
    DeformablePSROIAlign,
)
from tensorflow_addons.layers.deformable_conv2d import _deformable_conv2d_ops_so
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class DeformableConv2DTest(tf.test.TestCase):
    def _forward(
        self,
        input,
        filters,
        kernel_size=(3, 3),
        num_groups=1,
        deformable_groups=1,
        strides=(1, 1),
        im2col=1,
        use_bias=False,
        padding="valid",
        data_format="channels_last",
        dilations=(1, 1),
    ):
        input_op = tf.convert_to_tensor(input)
        output = DeformableConv2D(
            filters,
            kernel_size,
            num_groups,
            deformable_groups,
            strides,
            im2col,
            use_bias,
            padding,
            data_format,
            dilations,
        )(input_op)
        return output

    def _create_test_data(self, data_format):
        height = 20
        width = 20
        channel = 3
        batch = 1
        val = np.random.uniform(size=[batch, height, width, channel]).astype(np.float32)
        if data_format == "channels_first":
            val = np.transpose(val, [0, 3, 1, 2])
        return val

    """
    Because DeformableConv2D layer use built in random_normal initializer to initialize weight, So the output can't be actually tested, So here we just simple compare the result in tf.nn.conv2d and _deformable_conv2d function in deformable_conv2d.py.
    """

    def _forward_simple(self, data_format, use_gpu=False):
        with test_utils.device(False):
            batch_size = 4
            padding = "SAME"
            kernel_h = 3
            kernel_w = 3
            channel = 3
            height = 20
            width = 20
            out_channel = 16
            # input = tf.random.uniform(shape=[batch_size, channel, height, width], maxval=10)
            input = tf.convert_to_tensor(
                [i for i in range(batch_size * channel * height * width)],
                dtype=tf.float32,
            )
            input = tf.reshape(input, [batch_size, channel, height, width])
            input_trans = tf.transpose(input, [0, 2, 3, 1])
            # filter = tf.random.uniform(
            #        shape=[kernel_h, kernel_w, channel, out_channel], maxval=10
            #    )
            filter = tf.convert_to_tensor(
                np.random.uniform(
                    0, 1, [kernel_h, kernel_w, channel, out_channel]
                ).astype(np.float32)
            )
            filter_deform = tf.transpose(filter, [3, 2, 0, 1])
            offset = tf.constant(
                [
                    0.0
                    for i in range(
                        batch_size * kernel_h * kernel_w * 2 * height * width
                    )
                ],
                shape=[batch_size, kernel_h * kernel_w * 2, height, width],
            )
            mask = tf.constant(
                [1.0 for i in range(batch_size * kernel_h * kernel_w * height * width)],
                shape=[batch_size, kernel_h * kernel_w, height, width],
            )
            result1 = _deformable_conv2d_ops_so.ops.addons_deformable_conv2d(
                input=input,
                filter=filter_deform,
                offset=offset,
                mask=mask,
                strides=[1, 1, 1, 1],
                num_groups=1,
                deformable_groups=1,
                im2col_step=1,
                no_bias=True,
                padding=padding,
                data_format="NCHW",
                dilations=[1, 1, 1, 1],
            )
            result2 = tf.nn.conv2d(input_trans, filter, [1, 1, 1, 1], padding)
            result2 = tf.transpose(result2, [0, 3, 1, 2])
            # print("Debug!", tf.reduce_mean(result1 - result2))
            self.assertAllClose(result1, result2, 1e-4, 1e-4)

    """
    def _gradients(self, data_format, use_gpu=False):
        with test_utils.device(use_gpu):
            with tf.GradientTape(persistent=True) as tape:
                val = self._create_test_data(data_format)
                input = tf.constant(val, dtype=tf.float32)
                if data_format == "channels_last":
                    input = tf.transpose(input, [0, 3, 1, 2])
                input_trans = tf.transpose(input, [0, 2, 3, 1])
                padding = "SAME"
                kernel_h = 3
                kernel_w = 3
                channel = 3
                height = 20
                width = 20
                out_channel = 16
                filter = tf.Variable(
                    np.random.uniform(0, 1, [kernel_h, kernel_w, channel, out_channel]
                    ).astype(np.float32)
                )
                tape.watch(filter)
                filter_deform = tf.transpose(filter, [3, 2, 0, 1])
                offset = tf.constant(
                    [0.0 for i in range(kernel_h * kernel_w * 2 * height * width)],
                    shape=[1, kernel_h * kernel_w * 2, height, width],
                )
                mask = tf.constant(
                    [1.0 for i in range(kernel_h * kernel_w * height * width)],
                    shape=[1, kernel_h * kernel_w, height, width],
                )
                result1 = _deformable_conv2d_ops_so.ops.addons_deformable_conv2d(
                    input=input,
                    filter=filter_deform,
                    offset=offset,
                    mask=mask,
                    strides=[1, 1, 1, 1],
                    num_groups=1,
                    deformable_groups=1,
                    im2col_step=1,
                    no_bias=True,
                    padding=padding,
                    data_format="NCHW",
                    dilations=[1, 1, 1, 1],
                )
                result2 = tf.nn.conv2d(input_trans, filter, [1, 1, 1, 1], "SAME")
                grad1 = tape.gradient(result1, filter)
                grad2 = tape.gradient(result2, filter)
                self.assertAllClose(grad1, grad2, 1e-4, 1e-4)
    """

    def _keras(self, data_format, use_gpu=False):
        inputs = self._create_test_data(data_format)
        self._forward(inputs, 64, data_format=data_format)

    def testForwardNCHW(self):
        self._forward_simple(data_format="channels_first", use_gpu=False)
        if tf.test.is_gpu_available():
            self._forward_simple(data_format="channels_first", use_gpu=True)

    def testForwardNHWC(self):
        self._forward_simple(data_format="channels_last", use_gpu=False)
        if tf.test.is_gpu_available():
            self._forward_simple(data_format="channels_last", use_gpu=True)

    """		
    def testBackwardNCHW(self):
        self._gradients(data_format="channels_first", use_gpu=False)
        if tf.test.is_gpu_available():
            self._gradients(data_format="channels_first", use_gpu=True)

    def testBackwardNHWC(self):
        self._gradients(data_format="channels_last", use_gpu=False)
        if tf.test.is_gpu_available():
            self._gradients(data_format="channels_last", use_gpu=True)
    """

    def testKerasNCHW(self):
        self._keras(data_format="channels_first", use_gpu=False)
        if tf.test.is_gpu_available():
            self._keras(data_format="channels_first", use_gpu=True)

    def testKerasNHWC(self):
        self._keras(data_format="channels_last", use_gpu=False)
        if tf.test.is_gpu_available():
            self._keras(data_format="channels_last", use_gpu=True)


@test_utils.run_all_in_graph_and_eager_modes
class DeformablePSROIAlignTest(tf.test.TestCase):
    def _forward_simple(self, data_format, use_gpu=False):
        featuremap = tf.random.normal(shape=[1, 64, 100, 100])
        rois = tf.convert_to_tensor(
            [[0, 1, 1, 800, 800], [0, 2, 2, 400, 400]], dtype=tf.float32
        )
        spatial_scale = 1 / 16
        group_size = 1
        pooled_size = 7
        sample_per_part = 4
        part_size = 7
        trans_std = 1
        (
            offset_t,
            top_count,
        ) = _deformable_conv2d_ops_so.ops.addons_deformable_psroi_pool(
            featuremap,
            rois,
            tf.convert_to_tensor(0),
            pooled_size=pooled_size,
            no_trans=True,
            spatial_scale=spatial_scale,
            output_dim=64,
            group_size=group_size,
            part_size=part_size,
            sample_per_part=sample_per_part,
            trans_std=trans_std,
        )
        return offset_t

    def _keras(self, data_format, use_gpu=False):
        featuremap = tf.random.normal(shape=[1, 64, 100, 100])
        rois = tf.convert_to_tensor(
            [[0, 1, 1, 800, 800], [0, 2, 2, 400, 400]], dtype=tf.float32
        )
        psroilayer = DeformablePSROIAlign(output_dim=64, data_format="channels_first")
        ret = psroilayer([featuremap, rois])
        return ret

    def testKerasNCHW(self):
        self._keras(data_format="channels_first", use_gpu=False)
        if tf.test.is_gpu_available():
            self._keras(data_format="channels_first", use_gpu=True)

    def testKerasNHWC(self):
        self._keras(data_format="channels_last", use_gpu=False)
        if tf.test.is_gpu_available():
            self._keras(data_format="channels_last", use_gpu=True)


if __name__ == "__main__":
    tf.test.main()
