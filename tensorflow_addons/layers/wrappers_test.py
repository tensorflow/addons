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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers import wrappers
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class WeightNormalizationTest(tf.test.TestCase):
    def test_weightnorm(self):
        test_utils.layer_test(
            wrappers.WeightNormalization,
            kwargs={
                'layer': tf.keras.layers.Conv2D(5, (2, 2)),
            },
            input_shape=(2, 4, 4, 3))

    def _check_data_init(self, data_init, input_data, expected_output):
        layer = tf.keras.layers.Dense(
            input_data.shape[-1],
            activation=None,
            kernel_initializer='identity',
            bias_initializer='zeros')
        test_utils.layer_test(
            wrappers.WeightNormalization,
            kwargs={
                'layer': layer,
                'data_init': data_init,
            },
            input_data=input_data,
            expected_output=expected_output)

    def test_weightnorm_with_data_init_is_false(self):
        input_data = np.array([[[-4, -4], [4, 4]]], dtype=np.float32)
        self._check_data_init(
            data_init=False, input_data=input_data, expected_output=input_data)

    def test_weightnorm_with_data_init_is_true(self):
        input_data = np.array([[[-4, -4], [4, 4]]], dtype=np.float32)
        self._check_data_init(
            data_init=True,
            input_data=input_data,
            expected_output=input_data / 4)

    def test_weightnorm_non_layer(self):
        images = tf.random.uniform((2, 4, 43))
        with self.assertRaises(AssertionError):
            wrappers.WeightNormalization(images)

    def test_weightnorm_non_kernel_layer(self):
        images = tf.random.uniform((2, 2, 2))
        with self.assertRaisesRegexp(ValueError, 'contains a `kernel`'):
            non_kernel_layer = tf.keras.layers.MaxPooling2D(2, 2)
            wn_wrapper = wrappers.WeightNormalization(non_kernel_layer)
            wn_wrapper(images)

    def test_weightnorm_with_time_dist(self):
        batch_shape = (32, 16, 64, 64, 3)
        inputs = tf.keras.layers.Input(batch_shape=batch_shape)
        a = tf.keras.layers.Conv2D(3, 5)
        b = wrappers.WeightNormalization(a)
        out = tf.keras.layers.TimeDistributed(b)(inputs)
        model = tf.keras.Model(inputs, out)

    def test_save_file_h5(self):
        conv = tf.keras.layers.Conv1D(1, 1)
        wn_conv = wrappers.WeightNormalization(conv)
        model = tf.keras.Sequential(layers=[wn_conv])
        model.build([1, 2, 3])
        model.save_weights('/tmp/model.h5')

        import os
        os.remove('/tmp/model.h5')
        # TODO: Find a better way to test this


if __name__ == "__main__":
    tf.test.main()
