# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may noa use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.image import mean_filter2d
from tensorflow_addons.image import median_filter2d
from tensorflow_addons.utils import test_utils


class MeanFilter2dTest(tf.test.TestCase):
    def _validate_mean_filter2d(self,
                                  inputs,
                                  expected_values,
                                  filter_shape=(3, 3)):
        output = mean_filter2d(inputs, filter_shape)
        self.assertAllClose(output, expected_values)

    @test_utils.run_in_graph_and_eager_modes
    def test_filter_tuple(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        for filter_shape in [3, 3.5, 'dt', None]:
            with self.assertRaisesRegexp(TypeError,
                                         'Filter shape must be a tuple'):
                mean_filter2d(tf_img, filter_shape)

        filter_shape = (3, 3, 3)
        msg = ('Filter shape must be a tuple of 2 integers. '
               'Got %s values in tuple' % len(filter_shape))
        with self.assertRaisesRegexp(ValueError, msg):
            mean_filter2d(tf_img, filter_shape)

        msg = 'Size of the filter must be Integers'
        for filter_shape in [(3.5, 3), (None, 3)]:
            with self.assertRaisesRegexp(TypeError, msg):
                mean_filter2d(tf_img, filter_shape)

    @test_utils.run_in_graph_and_eager_modes
    def test_filter_value(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        with self.assertRaises(ValueError):
            mean_filter2d(tf_img, (4, 3))

    @test_utils.run_deprecated_v1
    def test_dimension(self):
        for image_shape in [(3, 4, None), (3, None, 4), (None, 3, 4)]:
            with self.assertRaises(TypeError):
                tf_img = tf.compat.v1.placeholder(tf.int32, shape=image_shape)
                mean_filter2d(tf_img)

    @test_utils.run_in_graph_and_eager_modes
    def test_image_vs_filter(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)
        filter_shape = (3, 5)
        with self.assertRaises(ValueError):
            mean_filter2d(tf_img, filter_shape)

    @test_utils.run_in_graph_and_eager_modes
    def test_three_channels(self):
        tf_img = [[[0.32801723, 0.08863795, 0.79119259],
                   [0.35526001, 0.79388736, 0.55435993],
                   [0.11607035, 0.55673079, 0.99473371]],
                  [[0.53240645, 0.74684819, 0.33700031],
                   [0.01760473, 0.28181609, 0.9751476],
                   [0.01605137, 0.8292904, 0.56405609]],
                  [[0.57215374, 0.10155051, 0.64836128],
                   [0.36533048, 0.91401874, 0.02524159],
                   [0.56379134, 0.9028874, 0.19505117]]]

        tf_img = tf.convert_to_tensor(value=tf_img)
        expt = [[[34, 54, 75], [38, 93, 119], [14, 69, 87]],
                [[61, 82, 94], [81, 147, 144], [40, 121, 93]],
                [[42, 57, 56], [58, 106, 77], [27, 82, 49]]]
        expt = tf.convert_to_tensor(value=expt)
        self._validate_mean_filter2d(tf_img, expt)

        
class MedianFilter2dTest(tf.test.TestCase):
    def _validate_median_filter2d(self,
                                  inputs,
                                  expected_values,
                                  filter_shape=(3, 3)):
        output = median_filter2d(inputs, filter_shape)
        self.assertAllClose(output, expected_values)

    @test_utils.run_in_graph_and_eager_modes
    def test_filter_tuple(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        for filter_shape in [3, 3.5, 'dt', None]:
            with self.assertRaisesRegexp(TypeError,
                                         'Filter shape must be a tuple'):
                median_filter2d(tf_img, filter_shape)

        filter_shape = (3, 3, 3)
        msg = ('Filter shape must be a tuple of 2 integers. '
               'Got %s values in tuple' % len(filter_shape))
        with self.assertRaisesRegexp(ValueError, msg):
            median_filter2d(tf_img, filter_shape)

        msg = 'Size of the filter must be Integers'
        for filter_shape in [(3.5, 3), (None, 3)]:
            with self.assertRaisesRegexp(TypeError, msg):
                median_filter2d(tf_img, filter_shape)

    @test_utils.run_in_graph_and_eager_modes
    def test_filter_value(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        with self.assertRaises(ValueError):
            median_filter2d(tf_img, (4, 3))

    @test_utils.run_deprecated_v1
    def test_dimension(self):
        for image_shape in [(3, 4, None), (3, None, 4), (None, 3, 4)]:
            with self.assertRaises(TypeError):
                tf_img = tf.compat.v1.placeholder(tf.int32, shape=image_shape)
                median_filter2d(tf_img)

    @test_utils.run_in_graph_and_eager_modes
    def test_image_vs_filter(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)
        filter_shape = (3, 5)
        with self.assertRaises(ValueError):
            median_filter2d(tf_img, filter_shape)

    @test_utils.run_in_graph_and_eager_modes
    def test_three_channels(self):
        tf_img = [[[0.32801723, 0.08863795, 0.79119259],
                   [0.35526001, 0.79388736, 0.55435993],
                   [0.11607035, 0.55673079, 0.99473371]],
                  [[0.53240645, 0.74684819, 0.33700031],
                   [0.01760473, 0.28181609, 0.9751476],
                   [0.01605137, 0.8292904, 0.56405609]],
                  [[0.57215374, 0.10155051, 0.64836128],
                   [0.36533048, 0.91401874, 0.02524159],
                   [0.56379134, 0.9028874, 0.19505117]]]

        tf_img = tf.convert_to_tensor(value=tf_img)
        expt = [[[0, 0, 0], [4, 71, 141], [0, 0, 0]],
                [[83, 25, 85], [90, 190, 143], [4, 141, 49]],
                [[0, 0, 0], [4, 71, 49], [0, 0, 0]]]
        expt = tf.convert_to_tensor(value=expt)
        self._validate_median_filter2d(tf_img, expt)


if __name__ == "__main__":
    tf.test.main()
