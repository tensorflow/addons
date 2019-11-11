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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.optical_flow import CorrelationCost
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class CorrelationCostTest(tf.test.TestCase):
    def _forward(self, input_a, input_b, kernel_size, max_displacement,
                 stride_1, stride_2, pad, data_format):

        input_a_op = tf.convert_to_tensor(input_a, dtype=tf.float32)
        input_b_op = tf.convert_to_tensor(input_b, dtype=tf.float32)

        output = CorrelationCost(
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride_1=stride_1,
            stride_2=stride_2,
            pad=pad,
            data_format=data_format)([input_a_op, input_b_op])

        return output

    def _create_test_data(self, data_format):
        # Produce test data for _forward_simple and _keras methods
        val_a = np.array(
            [[[[0, -6, 9, 5], [1, -5, 10, 3], [2, -4, 11, 1]],
              [[3, -3, 12, -1], [4, -2, 13, -3], [5, -1, 14, -5]]],
             [[[6, 0, 15, -7], [7, 1, 16, -9], [8, 2, 17, -11]],
              [[9, 3, 18, -13], [10, 4, 19, -15], [11, 5, 20, -17]]]],
            dtype=np.float32)

        # pylint: disable=too-many-function-args
        val_b = val_a.transpose(2, 3, 0, 1).reshape(2, 2, 3, 4)
        # pylint: enable=too-many-function-args

        if data_format == 'channels_last':
            val_a = np.moveaxis(val_a, 1, -1)
            val_b = np.moveaxis(val_b, 1, -1)

        return val_a, val_b

    def _forward_simple(self, data_format):
        # We are just testing where the output has vanishing values.
        with test_utils.use_gpu():
            val_a, val_b = self._create_test_data(data_format)
            input_a = tf.constant(val_a, dtype=tf.float32)
            input_b = tf.constant(val_b, dtype=tf.float32)

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

            if data_format == 'channels_last':
                # NHWC -> NCHW
                actual = tf.transpose(actual, [0, 3, 1, 2])

            # We can test fixed ids, as output is independent from data_format
            expected_ids = np.concatenate([np.zeros(464,), np.ones(464,)])
            self.assertAllClose(
                tf.where(tf.equal(actual, 0))[:, 0], expected_ids)

            counts = [54, 52, 54, 50, 44, 50, 54, 52, 54]
            expected_ids = np.concatenate(
                [k * np.ones(v,) for k, v in enumerate(counts)])
            expected_ids = np.concatenate([expected_ids, expected_ids])
            self.assertAllClose(
                tf.where(tf.equal(actual, 0))[:, 1], expected_ids)
            self.assertEqual(actual.shape, (2, 9, 7, 8))

    def _gradients(self, data_format):
        with test_utils.use_gpu():
            batch, channels, height, width = 2, 3, 5, 6
            input_a = np.random.randn(batch, channels, height,
                                      width).astype(np.float32)
            input_b = np.random.randn(batch, channels, height,
                                      width).astype(np.float32)

            kernel_size = 1
            max_displacement = 2
            stride_1 = 1
            stride_2 = 2
            pad = 4

            if data_format == 'channels_last':
                input_a = tf.transpose(input_a, [0, 2, 3, 1])
                input_b = tf.transpose(input_b, [0, 2, 3, 1])

            input_a_op = tf.convert_to_tensor(input_a)
            input_b_op = tf.convert_to_tensor(input_b)

            def correlation_fn(input_a, input_b):
                return CorrelationCost(
                    kernel_size=kernel_size,
                    max_displacement=max_displacement,
                    stride_1=stride_1,
                    stride_2=stride_2,
                    pad=pad,
                    data_format=data_format)([input_a, input_b])

            theoretical, numerical = tf.test.compute_gradient(
                correlation_fn, [input_a_op, input_b_op])

            self.assertAllClose(theoretical[0], numerical[0], atol=1e-3)

    def _keras(self, data_format):
        # Unable to use `layer_test` as this layer has multiple inputs.
        with test_utils.use_gpu():
            val_a, val_b = self._create_test_data(data_format)

            # yapf: disable
            input_a = tf.keras.Input(shape=val_a.shape[1:])
            input_b = tf.keras.Input(shape=val_b.shape[1:])

            layer = CorrelationCost(
                kernel_size=1,
                max_displacement=2,
                stride_1=1,
                stride_2=2,
                pad=4,
                data_format=data_format)

            expected_output_shape = tuple(
                layer.compute_output_shape([input_a.shape, input_b.shape]))
            # yapf: enable

            x = [input_a, input_b]
            y = layer(x)
            model = tf.keras.models.Model(x, y)
            actual_output = model([val_a, val_b])

            expected_output_type = 'float32'
            if tf.keras.backend.dtype(y[0]) != expected_output_type:
                raise AssertionError(
                    "Inferred output type %s does not equal "
                    "expected output type %s" % (tf.keras.backend.dtype(y[0]),
                                                 expected_output_type))

            if actual_output.shape[1:] != expected_output_shape[0][1:]:
                raise AssertionError(
                    "Expected shape %s does not equal output shape"
                    "%s" % (actual_output.shape, expected_output_shape[0]))

    def testForwardNCHW(self):
        self._forward_simple(data_format='channels_first')

    def testForwardNHWC(self):
        self._forward_simple(data_format='channels_last')

    def testBackwardNCHW(self):
        self._gradients(data_format='channels_first')

    def testBackwardNHWC(self):
        self._gradients(data_format='channels_last')

    def testKerasNCHW(self):
        self._keras(data_format='channels_first')

    def testKerasNHWC(self):
        self._keras(data_format='channels_last')


if __name__ == "__main__":
    tf.test.main()
