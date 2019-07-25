# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.correlation_cost import correlation_cost


class CorrelationCostTest(tf.test.TestCase):
    def _forward(self,
                 input_a,
                 input_b,
                 kernel_size,
                 max_displacement,
                 stride_1,
                 stride_2,
                 pad,
                 data_format):

        input_a_op = tf.convert_to_tensor(input_a, dtype=tf.float32)
        input_b_op = tf.convert_to_tensor(input_b, dtype=tf.float32)

        output = correlation_cost(
            input_a_op,
            input_b_op,
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride_1=stride_1,
            stride_2=stride_2,
            pad=pad,
            data_format=data_format)

        return output

    def _forward_simple(self, data_format='NCHW'):
        # cumbersome calculation by hand for a fixed input
        # we just test where zeros occurs and a few entries
        val = [[[[0, -6, 9, 5], [1, -5, 10, 3], [2, -4, 11, 1]],
                [[3, -3, 12, -1], [4, -2, 13, -3], [5, -1, 14, -5]]],
               [[[6, 0, 15, -7], [7, 1, 16, -9], [8, 2, 17, -11]],
                [[9, 3, 18, -13], [10, 4, 19, -15], [11, 5, 20, -17]]]]

        input_a = tf.constant(np.array(val), dtype=tf.float32)
        valb = np.array(val).transpose(2, 3, 0, 1).reshape(2, 2, 3, 4)
        input_b = tf.constant(valb, dtype=tf.float32)

        if data_format == 'NHWC':
            input_a = tf.transpose(input_a, [0, 2, 3, 1])
            input_b = tf.transpose(input_b, [0, 2, 3, 1])

        input_a_tensor = tf.convert_to_tensor(input_a, dtype=tf.float32)
        input_b_tensor = tf.convert_to_tensor(input_b, dtype=tf.float32)

        kernel_size = 1
        max_displacement = 2
        stride_1 = 1
        stride_2 = 2
        pad = 4

        actual = self._forward(
            input_a_tensor,
            input_b_tensor,
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride_1=stride_1,
            stride_2=stride_2,
            pad=pad,
            data_format=data_format)

        if data_format == 'NHWC':
            # NHWC -> NCHW
            actual = tf.transpose(actual, [0, 3, 1, 2])

        # we just need to test fixed ids, as output is NCHW independently from data_format
        expected_ids = np.concatenate([np.zeros(464,), np.ones(464,)])
        self.assertAllClose(np.where(actual.numpy() == 0)[0], expected_ids)

        counts = [54, 52, 54, 50, 44, 50, 54, 52, 54]
        expected_ids = np.concatenate(
            [k * np.ones(v,) for k, v in enumerate(counts)])
        expected_ids = np.concatenate([expected_ids, expected_ids])
        self.assertAllClose(np.where(actual.numpy() == 0)[1], expected_ids)
        self.assertEqual(actual.shape, (2, 9, 7, 8))

    def _gradients(self, data_format='NCHW'):

        batch, channels, height, width = 2, 3, 5, 6
        input_a = tf.random.normal([batch, channels, height, width])
        input_b = tf.random.normal([batch, channels, height, width])

        kernel_size = 1
        max_displacement = 2
        stride_1 = 1
        stride_2 = 2
        pad = 4

        if data_format == 'NHWC':
            input_a = tf.transpose(input_a, [0, 2, 3, 1])
            input_b = tf.transpose(input_b, [0, 2, 3, 1])

        input_a_op = tf.convert_to_tensor(input_a, dtype=tf.float32)
        input_b_op = tf.convert_to_tensor(input_b, dtype=tf.float32)

        def correlation_fn(inputs):
            output = correlation_cost(
                inputs[0],
                inputs[1],
                kernel_size=kernel_size,
                max_displacement=max_displacement,
                stride_1=stride_1,
                stride_2=stride_2,
                pad=pad,
                data_format=data_format)
            return output

        theoretical, numerical = tf.test.compute_gradient(
            correlation_fn, [[input_a_op, input_b_op]])

        self.assertAllClose(theoretical[0], numerical[0], atol=1e-3)

    def testForwardNCHW(self):
        self._forward_simple(data_format='NCHW')

    def testForwardNHWC(self):
        self._forward_simple(data_format='NHWC')

    def testBackwardNCHW(self):
        self._gradients(data_format='NCHW')

    def testBackwardNHWC(self):
        self._gradients(data_format='NHWC')


if __name__ == "__main__":
    tf.test.main()
